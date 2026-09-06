"""
Sector Exposure Screening engine.

Physical-evidence screening for investors, property owners and governments:
sector sensitivity x location hazards x physical trajectory over time, with an
official-statistics crime layer where one openly exists.

Honesty contract (HARD):
    - This module is NOT investment advice and never says "invest / don't invest".
    - It never predicts losses or computes financial metrics.
    - Every unavailable component is a declared gap; nothing is invented.
    - Crime data comes from official open sources only (data.police.uk).
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evidence import content_hash
from .ontology import ClaimStatus, Confidence
from .verification import VERIFICATION_HAZARDS, verify_asset

# Upstream fetchers
from ..dashboard.crime_stats import fetch_street_crime
from ..dashboard.population import fetch_population
from ..dashboard.real_data import _UA, _valid_point, fetch_climate_series
from ..gis_mapping.forest_loss import fetch_forest_loss

ROOT = Path(__file__).resolve().parents[2]
_KB_PATH = ROOT / "config" / "sector_profiles.json"

_OHSOME_URL = "https://api.ohsome.org/v1/elements/count"
_BUILDING_FILTER = "building=*"
_EPOCH_2015 = "2015-01-01"
_OHSOME_RADIUS_M = 500

DISCLAIMER = (
    "Physical-risk screening evidence only. This is not investment advice, "
    "not a valuation, and not a prediction. Crime figures come from official "
    "statistics only where an open official source exists; financial metrics "
    "are not quantified."
)

HONESTY_CONTRACT = (
    "Unavailable data is declared as a gap; no hazard level, trajectory value, "
    "crime figure or financial metric is invented."
)

_WEIGHT_SCORES = {"high": 2, "medium": 1, "low": 1}

# Band thresholds for the deterministic screening-exposure score.
# Documented as declared screening heuristics, not validated risk bands.
_BANDS = [
    (0, 4, "lower"),
    (5, 12, "moderate"),
    (13, 24, "elevated"),
]

def _load_kb() -> Dict[str, Any]:
    with open(_KB_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _sector_ids() -> List[str]:
    return [s["id"] for s in _load_kb().get("sectors", [])]

def _level_score(level_label: Optional[str]) -> int:
    if not level_label:
        return 0
    label = level_label.strip().lower()
    if label.startswith("extreme"):
        return 4
    if label.startswith("severe"):
        return 3
    if label.startswith("moderate"):
        return 2
    if label.startswith("mild") or label.startswith("low"):
        return 1
    return 0

def _exposure_band(score: int) -> str:
    for low, high, band in _BANDS:
        if low <= score <= high:
            return band
    return "high"

def _fetch_building_epoch_counts(
    lat: float, lon: float, radius_m: int = _OHSOME_RADIUS_M
) -> Dict[str, Any]:
    """
    Count mapped buildings at the same point for 2015 and the latest ohsome
    extract. Returns growth_pct only when both counts are available and the
    2015 count is non-zero.

    ohsome note: the 2015 epoch uses an explicit ``time``; the "latest"
    epoch OMITS it — the API then returns the most recent snapshot, while an
    explicit near-future date can 404 (live-checked).
    """
    queries = [
        ("epoch_2015", {"time": _EPOCH_2015}),
        ("latest", {}),
    ]
    counts: Dict[str, Optional[int]] = {}
    timestamps: Dict[str, Optional[str]] = {}
    errors: List[str] = []

    for label, extra in queries:
        body = urllib.parse.urlencode({
            "bcircles": f"{lon},{lat},{radius_m}",
            "filter": _BUILDING_FILTER,
            **extra,
        }).encode("utf-8")
        req = urllib.request.Request(
            _OHSOME_URL, data=body, headers={"User-Agent": _UA}
        )
        try:
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            result = (data.get("result") or [{}])[0]
            counts[label] = int(result.get("value") or 0)
            timestamps[label] = result.get("timestamp")
        except Exception as exc:
            counts[label] = None
            errors.append(f"{label}: {exc}")

    epoch_2015 = counts.get("epoch_2015")
    latest = counts.get("latest")
    growth_pct: Optional[float] = None
    if epoch_2015 is not None and latest is not None and epoch_2015 > 0:
        growth_pct = round(100.0 * (latest - epoch_2015) / epoch_2015, 1)

    return {
        "epoch_2015": epoch_2015,
        "latest": latest,
        "epoch_2015_timestamp": timestamps.get("epoch_2015"),
        "latest_timestamp": timestamps.get("latest"),
        "growth_pct": growth_pct,
        "radius_m": radius_m,
        "source": "OpenStreetMap via ohsome API (Heidelberg Institute)",
        "note": "Mapped building counts; OSM completeness varies by region and epoch.",
        "errors": errors if errors else None,
    }

def _build_trajectory(lat: float, lon: float) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    gaps: List[Dict[str, str]] = []
    trajectory: Dict[str, Any] = {}

    # Climate reanalysis
    climate = fetch_climate_series(lat, lon)
    if climate.get("error"):
        gaps.append({"component": "climate_trajectory", "reason": climate["error"]})
        trajectory["climate"] = {"claim_status": ClaimStatus.UNKNOWN.value, "reason": climate["error"]}
    else:
        current = climate.get("current") or {}
        trajectory["climate"] = {
            "mean_tmax_anomaly_c": current.get("mean_tmax_anomaly_c"),
            "precip_pct_of_baseline": current.get("precip_pct_of_baseline"),
            "baseline_period": climate.get("baseline", {}).get("period"),
            "latest_complete_year": current.get("year"),
            "source": climate.get("source"),
            "claim_status": ClaimStatus.MODELLED.value,
            "confidence": Confidence.MEDIUM.value,
            "note": "Reanalysis-derived anomaly versus the stated baseline period.",
        }

    # Forest cover change
    forest = fetch_forest_loss(lat, lon)
    if forest.get("error"):
        gaps.append({"component": "forest_loss", "reason": forest["error"]})
        trajectory["forest"] = {"claim_status": ClaimStatus.UNKNOWN.value, "reason": forest["error"]}
    else:
        loss_years = forest.get("loss_years") or {}
        trajectory["forest"] = {
            "tree_cover_2000_mean_pct": forest.get("tree_cover_2000_mean_pct"),
            "forested_fraction_2000": forest.get("forested_fraction_2000"),
            "loss_detected": forest.get("loss_detected"),
            "loss_years": loss_years,
            "loss_after_2020": forest.get("loss_after_2020"),
            "source": forest.get("source"),
            "claim_status": ClaimStatus.OBSERVED.value,
            "confidence": Confidence.HIGH.value,
            "vintage_note": forest.get("vintage_note"),
            "limitations": forest.get("limitations"),
        }

    # Urban expansion (OSM building counts via ohsome)
    buildings = _fetch_building_epoch_counts(lat, lon)
    if buildings.get("errors") and buildings.get("latest") is None:
        reason = "; ".join(buildings["errors"])
        gaps.append({"component": "urban_expansion", "reason": reason})
        trajectory["urban_expansion"] = {
            "claim_status": ClaimStatus.UNKNOWN.value,
            "reason": reason,
        }
    else:
        trajectory["urban_expansion"] = {
            "epoch_2015": buildings.get("epoch_2015"),
            "latest": buildings.get("latest"),
            "latest_date": buildings.get("latest_date"),
            "growth_pct": buildings.get("growth_pct"),
            "radius_m": buildings.get("radius_m"),
            "source": buildings.get("source"),
            "claim_status": ClaimStatus.OBSERVED.value,
            "confidence": Confidence.MEDIUM.value,
            "note": buildings.get("note"),
            "limitations": buildings.get("errors"),
        }

    # Population estimate
    pop = fetch_population(lat, lon, radius_km=3.0)
    if pop.get("error"):
        gaps.append({"component": "population", "reason": pop["error"]})
        trajectory["population"] = {
            "claim_status": ClaimStatus.UNKNOWN.value,
            "reason": pop["error"],
        }
    else:
        trajectory["population"] = {
            "estimated_population": pop.get("estimated_population"),
            "reference_year": pop.get("reference_year"),
            "radius_km": pop.get("radius_km"),
            "source": pop.get("source"),
            "claim_status": ClaimStatus.MODELLED.value,
            "confidence": Confidence.MEDIUM.value,
            "note": pop.get("estimate_note"),
        }

    trajectory["trend_note"] = _build_trend_note(trajectory)
    return trajectory, gaps

def _fmt_num(value: float) -> str:
    """Format a number without unnecessary trailing zeros."""
    if value == int(value):
        return str(int(value))
    return str(round(value, 1))

def _build_trend_note(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    climate = trajectory.get("climate") or {}
    if climate.get("mean_tmax_anomaly_c") is not None:
        sign = "+" if climate["mean_tmax_anomaly_c"] >= 0 else ""
        parts.append(
            f"Temperature anomaly {sign}{_fmt_num(climate['mean_tmax_anomaly_c'])} °C "
            f"vs {climate.get('baseline_period', 'baseline')}"
        )
    if climate.get("precip_pct_of_baseline") is not None:
        parts.append(
            f"precipitation {_fmt_num(climate['precip_pct_of_baseline'])}% of baseline"
        )

    forest = trajectory.get("forest") or {}
    loss_years = forest.get("loss_years") or {}
    if loss_years:
        years = sorted(loss_years.keys())
        if len(years) == 1:
            parts.append(f"tree-cover loss recorded in {years[0]}")
        else:
            parts.append(f"tree-cover loss across {years[0]}–{years[-1]}")
    elif forest.get("loss_detected") is False:
        parts.append("no tree-cover loss detected in the screened window")

    urban = trajectory.get("urban_expansion") or {}
    if urban.get("growth_pct") is not None:
        sign = "+" if urban["growth_pct"] >= 0 else ""
        parts.append(f"built-up features {sign}{_fmt_num(urban['growth_pct'])}% since 2015")
    elif urban.get("latest") is not None and urban.get("epoch_2015") is not None:
        parts.append("built-up feature count stable since 2015")

    pop = trajectory.get("population") or {}
    if pop.get("estimated_population") is not None:
        parts.append(
            f"population estimate {pop['estimated_population']:,.0f} "
            f"(WorldPop reference year {pop.get('reference_year')})"
        )

    if not parts:
        return "No trajectory components could be assembled for this location."
    note = "; ".join(parts) + "."
    return note[0].upper() + note[1:]

def _build_crime_block(lat: float, lon: float) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    crime = fetch_street_crime(lat, lon)
    if crime.get("jurisdiction_gap") or crime.get("claim_status") == ClaimStatus.UNKNOWN.value:
        return crime, {
            "component": "crime",
            "reason": crime.get("reason") or "Official crime statistics unavailable for this jurisdiction.",
        }
    return crime, None

def build_sector_screen(
    lat: float,
    lon: float,
    sectors: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a sector exposure screen for a point.

    Returns a deterministic, source-attached screening record. No financial
    metrics, no investment verdicts, and every missing component is declared.
    """
    if not _valid_point(lat, lon):
        raise ValueError("lat/lon out of range")

    kb = _load_kb()
    all_sectors = kb.get("sectors", [])
    requested = set(sectors) if sectors else None
    active = [s for s in all_sectors if requested is None or s["id"] in requested]

    # Run the shared hazard verification once.
    verification = verify_asset(lat, lon, name=name)
    checks_by_hazard = {c["hazard"]: c for c in verification.get("hazard_checks", [])}

    declared_gaps: List[Dict[str, str]] = []
    sector_results: List[Dict[str, Any]] = []

    for sector in active:
        exposures: List[Dict[str, Any]] = []
        score = 0
        for sh in sector.get("sensitive_hazards", []):
            hazard_id = sh["hazard"]
            check = checks_by_hazard.get(hazard_id)
            weight = sh.get("weight", "medium")
            weight_score = _WEIGHT_SCORES.get(weight, 1)
            rationale = sh.get("rationale", "")

            if check is None or check.get("claim_status") == ClaimStatus.UNKNOWN.value:
                level_label = None
                level_score = 0
                claim_status = ClaimStatus.UNKNOWN.value
                confidence = Confidence.LOW.value
                declared_gaps.append({
                    "sector": sector["id"],
                    "hazard": hazard_id,
                    "reason": (check or {}).get("limitations", [f"{hazard_id} data unavailable"])[0],
                })
            else:
                level_label = (check.get("level") or {}).get("label")
                level_score = _level_score(level_label)
                claim_status = check.get("claim_status", ClaimStatus.UNKNOWN.value)
                confidence = check.get("confidence", Confidence.MEDIUM.value)

            score += weight_score * level_score
            exposures.append({
                "hazard": hazard_id,
                "weight": weight,
                "level_label": level_label,
                "claim_status": claim_status,
                "confidence": confidence,
                "rationale": rationale,
            })

        sector_results.append({
            "id": sector["id"],
            "label": sector.get("label", sector["id"]),
            "hazard_exposures": exposures,
            "screening_exposure": {
                "score": score,
                "band": _exposure_band(score),
                "note": "screening indicator, not a validated model",
            },
            "investor_note": sector.get("investor_note", ""),
        })

    # Trajectory and crime are independent of sector selection.
    trajectory, traj_gaps = _build_trajectory(lat, lon)
    declared_gaps.extend(traj_gaps)

    crime_block, crime_gap = _build_crime_block(lat, lon)
    if crime_gap:
        declared_gaps.append(crime_gap)

    screen_id = content_hash({
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "sectors": sorted(s["id"] for s in active),
        "verification": verification.get("verification_id"),
    })[:16]

    return {
        "screen_id": screen_id,
        "location": {
            "lat": lat,
            "lon": lon,
            "name": name,
        },
        "sectors": sector_results,
        "trajectory": trajectory,
        "crime": crime_block,
        "declared_gaps": declared_gaps,
        "disclaimer": DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
        "methodology_note": kb.get("methodology_note", ""),
    }
