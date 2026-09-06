"""
Proactive protection recommendations and the automation framework.

Two layers, both generated from the *actual* detected conditions of the
current analysis — a recommendation is only emitted when its triggering
condition is really present, and it quotes the real values as evidence:

    build_recommendations(analysis)  -> preventive recommendations
        (WHAT / WHY / PRIORITY / EVIDENCE / EXPECTED EFFECT / DATA SOURCES)

    build_action_plan(analysis, recommendations, ops_config) -> action plan
        for the automation framework. Actions are clearly typed:
        - "automated"   : internal, no external contact (still not armed
                          unless explicitly enabled in config/operations.json)
        - "recommended" : requires an operational configuration (contacts,
                          responsibilities) before anything real happens.
        No real person or organization is ever contacted by default.
        Observed outcomes are only recorded after real execution; until
        then the outcome field is null (UNKNOWN).

Neither layer claims that an intervention guarantees prevention.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .cache import default_cache

_DEFAULT_OPS_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "operations.json"
)

NO_GUARANTEE_NOTE = (
    "Preventive measures reduce exposure and improve preparedness; no "
    "intervention guarantees that a wildfire will not start or spread."
)


# --------------------------------------------------------------------------
# Proactive recommendations
# --------------------------------------------------------------------------

def build_recommendations(analysis: Dict) -> List[Dict]:
    """
    Generate evidence-linked preventive recommendations from real analysis
    outputs. Each rule fires only when its condition is actually met.
    """
    fire_danger = analysis.get("fire_danger") or {}
    a = analysis.get("analysis") or {}
    weather = analysis.get("weather") or {}
    terrain = analysis.get("terrain") or {}
    landcover = analysis.get("landcover") or {}
    fires = analysis.get("active_fires") or {}
    trend = analysis.get("fire_danger_trend") or {}

    fwi = fire_danger.get("fwi") if fire_danger.get("available") else None
    fmc = a.get("fuel_moisture_baseline_pct")
    wind = weather.get("wind_speed_kmh")
    slope = terrain.get("slope_degrees")
    burnable = landcover.get("burnable", True) if "error" not in landcover else True
    risk = (a.get("risk") or {}).get("baseline")

    recs: List[Dict] = []

    def _add(rid, what, why, priority, evidence, effect, sources):
        recs.append({
            "id": rid,
            "kind": "preventive_recommendation",
            "what": what,
            "why": why,
            "priority": priority,
            "evidence": evidence,
            "expected_effect": effect,
            "data_sources": sources,
        })

    fwi_sources = ["Canadian FWI System (Open-Meteo daily data)"]
    fmc_sources = ["Sentinel-2 NDMI / Open-Meteo soil moisture"]

    if fwi is not None and fwi >= 38.0:
        _add(
            "fwi-high",
            "Increase monitoring frequency and preparedness level.",
            f"Fire-weather danger is high (FWI {fwi:.1f}, class "
            f"'{fire_danger.get('class')}').",
            "high" if fwi < 50.0 else "critical",
            {"driver": "fwi", "value": fwi, "threshold": 38.0},
            "Earlier detection and faster initial response during high-danger "
            "conditions.",
            fwi_sources,
        )

    if fmc is not None and fmc < 12.0:
        _add(
            "fuel-very-dry",
            "Inspect and reduce combustible vegetation in priority areas; "
            "consider targeted pre-hydration of fuel corridors.",
            f"Fuel moisture is very low ({fmc:.1f}% FMC) — fuels ignite and "
            "carry fire readily.",
            "high",
            {"driver": "fuel_moisture", "value": fmc, "threshold": 12.0},
            "Reduced ignition probability and slower fire spread in treated "
            "zones.",
            fmc_sources,
        )
    elif fmc is not None and fmc < 18.0:
        _add(
            "fuel-dry",
            "Monitor fuel dryness and plan targeted hydration for the most "
            "exposed zones.",
            f"Fuel moisture is below normal ({fmc:.1f}% FMC).",
            "moderate",
            {"driver": "fuel_moisture", "value": fmc, "threshold": 18.0},
            "Maintained fuel-moisture margin before conditions deteriorate.",
            fmc_sources,
        )

    if wind is not None and fmc is not None and wind >= 25.0 and fmc < 18.0:
        _add(
            "wind-dry",
            "Prioritize exposed zones and critical infrastructure on the "
            "downwind side; review ember/spotting exposure.",
            f"Strong wind ({wind:.0f} km/h) combined with dry fuel "
            f"({fmc:.1f}% FMC) favours fast, wind-driven spread.",
            "high",
            {"driver": "wind+fuel", "value": {"wind_kmh": wind, "fmc_pct": fmc},
             "threshold": {"wind_kmh": 25.0, "fmc_pct": 18.0}},
            "Protection effort concentrated where spread would be fastest.",
            ["Open-Meteo current weather"] + fmc_sources,
        )

    if slope is not None and slope >= 12.0:
        _add(
            "terrain-steep",
            "Inspect access routes and the firebreak strategy for steep "
            "terrain.",
            f"The terrain is steep ({slope:.1f}°) — fires spread significantly "
            "faster uphill and access is harder.",
            "moderate",
            {"driver": "slope", "value": slope, "threshold": 12.0},
            "Workable access and effective firebreak placement where spread "
            "accelerates.",
            ["DEM (OpenTopoData)"],
        )

    if trend.get("trend") == "rising" and fwi is not None and fwi >= 21.3:
        slope_pd = trend.get("slope_per_day")
        _add(
            "trend-rising",
            "Prepare for worsening conditions over the coming days; "
            "pre-position resources.",
            f"Fire danger is rising (FWI trend +{slope_pd}/day over the last "
            "days).",
            "moderate" if fwi < 38.0 else "high",
            {"driver": "fwi_trend", "value": slope_pd, "threshold": 0.5},
            "Readiness ahead of the danger peak instead of reactive response.",
            fwi_sources,
        )

    if fires.get("available") and (fires.get("count") or 0) > 0:
        _add(
            "active-fires",
            "Maintain heightened situational awareness and verify suppression "
            "readiness — active fires were detected nearby.",
            f"{fires['count']} active-fire detection(s) within "
            f"{fires.get('radius_km')} km in the last {fires.get('days')} days.",
            "critical",
            {"driver": "active_fires", "value": fires["count"], "threshold": 1},
            "Immediate awareness of ignitions in the surrounding area.",
            ["NASA FIRMS (VIIRS)"],
        )

    if risk is not None and risk >= 65.0 and burnable:
        _add(
            "risk-high",
            "Activate protection measures for critical assets (water "
            "pre-positioning, access checks, communication readiness).",
            f"The composite risk score is high ({risk:.1f}/100) over burnable "
            "land cover.",
            "high" if risk < 80.0 else "critical",
            {"driver": "risk_score", "value": risk, "threshold": 65.0},
            "Reduced exposure of people and critical infrastructure.",
            ["Talaix composite risk score"],
        )

    # ---- Environmental / ecological rules ------------------------------
    ecology = analysis.get("ecology") or {}
    exposure = analysis.get("exposure") or {}
    site = ecology.get("site_conditions") or {}
    moisture_regime = site.get("moisture_regime")
    lc_fraction = landcover.get("dominant_fraction")

    if risk is not None and risk >= 65.0 and burnable and lc_fraction is not None and lc_fraction >= 0.6:
        _add(
            "fuel-break",
            "Establish or maintain fuel breaks interrupting the continuous "
            "burnable cover, prioritizing asset edges.",
            f"Burnable land cover is dominant and continuous "
            f"({lc_fraction:.0%} '{landcover.get('dominant_label')}') with a "
            f"high composite risk ({risk:.1f}/100).",
            "high",
            {"driver": "land_cover+risk",
             "value": {"dominant_fraction": lc_fraction, "risk": risk},
             "threshold": {"dominant_fraction": 0.6, "risk": 65.0}},
            "Interrupted fuel continuity slows or stops surface-fire spread "
            "at treated lines.",
            ["ESA WorldCover", "Talaix composite risk score"],
        )

    if moisture_regime == "dry" and burnable:
        dryness_evidence = []
        if site.get("fuel_moisture_pct") is not None:
            dryness_evidence.append(f"FMC {site['fuel_moisture_pct']}%")
        if site.get("soil_moisture_m3m3") is not None:
            dryness_evidence.append(f"soil moisture {site['soil_moisture_m3m3']} m³/m³")
        if site.get("recent_precip_mm") is not None:
            dryness_evidence.append(f"recent rain {site['recent_precip_mm']} mm")
        _add(
            "ecological-restoration",
            "Plan ecological restoration toward less flammable native "
            "vegetation (see Environmental solutions for site-fitted species).",
            "A dry moisture regime is detected (" +
            ", ".join(dryness_evidence) + ") over burnable cover.",
            "moderate",
            {"driver": "moisture_regime", "value": moisture_regime,
             "threshold": "dry"},
            "Long-term shift to a less fire-prone, drought-adapted fuel "
            "structure and improved landscape moisture retention.",
            ["Talaix ecology engine", "Sentinel-2 NDMI / soil moisture"],
        )
        _add(
            "drought-preparedness",
            "Drought preparedness: secure water reserves and review irrigation "
            "/ hydration plans for priority zones.",
            f"Current conditions indicate drought stress (moisture regime "
            f"'{moisture_regime}').",
            "high" if (risk or 0) >= 65 else "moderate",
            {"driver": "moisture_regime", "value": moisture_regime,
             "threshold": "dry"},
            "Water availability is secured before the danger peak.",
            ["Open-Meteo soil moisture", "Sentinel-2 NDMI"],
        )

    if slope is not None and slope >= 15.0 and (risk or 0) >= 45.0:
        _add(
            "erosion-slope",
            "Include erosion and slope protection in planning; after any fire, "
            "prioritize slope stabilization and soil cover recovery.",
            f"Slopes are steep ({slope:.1f}°): post-fire erosion risk is "
            "elevated on burned steep terrain.",
            "moderate",
            {"driver": "slope", "value": slope, "threshold": 15.0},
            "Reduced post-fire soil loss, debris-flow and downstream damage.",
            ["DEM (OpenTopoData)"],
        )

    if fires.get("available") and (fires.get("count") or 0) > 0:
        _add(
            "post-fire-restoration",
            "If fire affects the area: plan post-fire ecological restoration "
            "(see Environmental solutions) and habitat recovery.",
            f"{fires['count']} active-fire detection(s) nearby — burned-area "
            "recovery planning may become necessary.",
            "moderate",
            {"driver": "active_fires", "value": fires["count"], "threshold": 1},
            "Faster ecosystem recovery and reduced erosion/invasion after fire.",
            ["NASA FIRMS (VIIRS)", "Talaix ecology engine"],
        )

    # ---- Exposure-driven rules (real OSM context) ----------------------
    if exposure.get("status") == "ok":
        va = (exposure.get("vulnerable_assets") or {})
        if (va.get("total") or 0) > 0 and (risk or 0) >= 65.0:
            details = ", ".join(
                f"{v} {k.replace('_', ' ')}" for k, v in va.items()
                if k != "total" and v
            )
            _add(
                "protect-critical-facilities",
                "Prioritize protection of the mapped critical facilities "
                "(defensible space, access, water points).",
                f"Critical facilities are mapped within {exposure.get('radius_m')} m "
                f"({details}) and the risk is high ({risk:.1f}/100).",
                "critical" if (risk or 0) >= 80 else "high",
                {"driver": "vulnerable_assets+risk",
                 "value": {"assets": va.get("total"), "risk": risk},
                 "threshold": {"assets": 1, "risk": 65.0}},
                "Reduced risk to the most vulnerable mapped assets first.",
                ["OpenStreetMap (Overpass)", "Talaix composite risk score"],
            )
        access = exposure.get("access") or {}
        if access.get("limited") and (risk or 0) >= 65.0:
            _add(
                "access-routes",
                "Review and clear access/evacuation routes; pre-identify "
                "alternative access for response crews.",
                "Access constraints detected: " +
                "; ".join(access.get("constraints") or []) +
                f" — with high risk ({risk:.1f}/100).",
                "high",
                {"driver": "access+risk",
                 "value": {"constraints": access.get("constraints"), "risk": risk},
                 "threshold": {"risk": 65.0}},
                "Response crews and residents retain workable routes under "
                "high danger.",
                ["OpenStreetMap (Overpass)", "DEM (OpenTopoData)"],
            )

    priority_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 4))
    return recs


# --------------------------------------------------------------------------
# Automation audit trail
# --------------------------------------------------------------------------

class PlanAuditStore:
    """
    SQLite audit trail for generated action plans.

    Every generated plan (automated or recommended) is recorded with its
    trigger values, actions and configuration state, so there is always a
    trace of WHAT was planned, WHEN, WHY — and whether anything was armed.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_cache().db_path
        self._lock = threading.Lock()
        with self._lock, sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_plan_audit (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    location TEXT,
                    level TEXT NOT NULL,
                    automation_enabled INTEGER NOT NULL,
                    trigger_summary TEXT NOT NULL,
                    actions TEXT NOT NULL
                )
                """
            )

    def record(self, location: str, level: str, automation_enabled: bool,
               trigger_summary: Dict, actions: List[Dict]) -> str:
        plan_id = uuid.uuid4().hex[:16]
        with self._lock, sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                "INSERT INTO action_plan_audit"
                " (id, created_at, location, level, automation_enabled,"
                "  trigger_summary, actions) VALUES (?,?,?,?,?,?,?)",
                (plan_id, datetime.utcnow().isoformat() + "Z", location, level,
                 1 if automation_enabled else 0,
                 json.dumps(trigger_summary, default=str),
                 json.dumps(actions, default=str)),
            )
        return plan_id

    def list(self, limit: int = 50) -> List[Dict]:
        with self._lock, sqlite3.connect(self.db_path, timeout=10.0) as conn:
            rows = conn.execute(
                "SELECT id, created_at, location, level, automation_enabled,"
                " trigger_summary, actions FROM action_plan_audit"
                " ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [
            {
                "id": r[0], "created_at": r[1], "location": r[2], "level": r[3],
                "automation_enabled": bool(r[4]),
                "trigger_summary": json.loads(r[5]), "actions": json.loads(r[6]),
            }
            for r in rows
        ]


# --------------------------------------------------------------------------
# Automation framework — action-plan generation
# --------------------------------------------------------------------------

def load_operations_config(path: Optional[str] = None) -> Dict:
    """
    Load the operational configuration (contacts, responsibilities).

    Default: automation disabled and no contacts — nothing external happens.
    """
    cfg_path = path or os.environ.get("HYDRASHIELD_OPERATIONS_CONFIG") or _DEFAULT_OPS_CONFIG
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        cfg = {}
    cfg.setdefault("enabled", False)
    cfg.setdefault("contacts", {})
    return cfg


def build_action_plan(
    analysis: Dict,
    recommendations: List[Dict],
    ops_config: Optional[Dict] = None,
    audit_store: Optional[PlanAuditStore] = None,
    record: bool = True,
) -> Dict:
    """
    Generate an action plan for the automation framework.

    The plan is always generated (it is information), but every external
    action is marked ``requires_operational_configuration`` unless an
    explicit, enabled operations config names a responsible contact. No
    notifications are sent by generating a plan.
    """
    cfg = ops_config if ops_config is not None else load_operations_config()
    contacts = cfg.get("contacts") or {}
    automation_enabled = bool(cfg.get("enabled"))

    a = analysis.get("analysis") or {}
    fire_danger = analysis.get("fire_danger") or {}
    risk = (a.get("risk") or {}).get("baseline")
    risk_class = (a.get("risk") or {}).get("class")
    fwi = fire_danger.get("fwi") if fire_danger.get("available") else None
    fmc = a.get("fuel_moisture_baseline_pct")
    wind = (analysis.get("weather") or {}).get("wind_speed_kmh")

    # Severity ladder from the real risk class.
    level = {"Extreme": "escalate", "High": "activate",
             "Moderate": "prepare", "Low": "routine"}.get(risk_class, "monitor")

    actions: List[Dict] = []

    def _auto(aid, action, trigger, note):
        actions.append({
            "id": aid,
            "action": action,
            "type": "automated",
            "status": "armed" if automation_enabled else "available_not_armed",
            "trigger": trigger,
            "outcome": None,  # OBSERVED OUTCOME only exists after real execution
            "note": note,
        })

    def _external(aid, action, contact_key, trigger):
        contact = contacts.get(contact_key)
        actions.append({
            "id": aid,
            "action": action,
            "type": "recommended",
            "status": "configured" if (automation_enabled and contact) else
                      "requires_operational_configuration",
            "trigger": trigger,
            "responsible": contact,
            "outcome": None,
            "note": None if (automation_enabled and contact) else
                    "No contact configured — nothing is sent.",
        })

    _auto(
        "monitor-frequency",
        "Increase monitoring frequency for this location (shorter watch "
        "re-check interval).",
        {"risk_class": risk_class, "fwi": fwi},
        "Internal scheduling change only; no external effect.",
    )

    if level in ("activate", "escalate"):
        _external(
            "notify-municipality",
            "Notify the responsible municipality / civil-protection contact "
            "with the current risk report.",
            "municipality",
            {"risk_class": risk_class, "risk": risk},
        )
        _external(
            "field-inspection",
            "Request a field inspection of priority zones and critical "
            "infrastructure.",
            "field_team",
            {"risk_class": risk_class, "fwi": fwi, "fuel_moisture_pct": fmc},
        )
        _auto(
            "water-check",
            "Verify water availability for the planned intervention volume "
            "and flag shortages.",
            {"intervention_water_m3": (analysis.get("methodology") or {})
             .get("intervention_water_m3")},
            "Internal check against configured water sources when enabled.",
        )

    if level == "escalate":
        if wind is not None and fmc is not None and wind >= 25.0 and fmc < 18.0:
            _external(
                "contractor-work-order",
                "Generate a vegetation-management work order for designated "
                "contractors (wind-driven spread exposure).",
                "vegetation_contractor",
                {"wind_kmh": wind, "fuel_moisture_pct": fmc},
            )
        _external(
            "escalate-persistent",
            "Escalate to the regional coordination contact if the risk "
            "remains Extreme at the next check.",
            "regional_coordination",
            {"risk_class": risk_class, "condition": "persists_at_next_check"},
        )

    plan = {
        "level": level,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trigger_summary": {
            "risk": risk, "risk_class": risk_class, "fwi": fwi,
            "fuel_moisture_pct": fmc, "wind_kmh": wind,
        },
        "actions": actions,
        "automation_enabled": automation_enabled,
        "linked_recommendations": [r["id"] for r in recommendations],
        "honesty_note": (
            "This plan is generated from real detected conditions. "
            "'recommended' actions require an operational configuration "
            "(config/operations.json) before anything is sent; no person or "
            "organization is contacted by default. Outcomes stay null "
            "(unknown) until an action is really executed and observed."
        ),
        "no_guarantee_note": NO_GUARANTEE_NOTE,
    }

    # Audit trail: every generated plan is recorded.
    if record:
        try:
            store = audit_store or PlanAuditStore()
            loc = analysis.get("location") or {}
            plan["audit_id"] = store.record(
                location=str(loc.get("name") or ""),
                level=level,
                automation_enabled=automation_enabled,
                trigger_summary=plan["trigger_summary"],
                actions=[{"id": a["id"], "type": a["type"], "status": a["status"]}
                         for a in actions],
            )
        except Exception:
            plan["audit_id"] = None  # auditing must never break the analysis

    return plan
