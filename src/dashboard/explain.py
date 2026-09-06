"""
"Why this score?" — transparent explanation of the Talaix risk score.

The composite score (0-100) is NOT a probability of fire. This module
breaks it down into the main contributing factors, each computed from the
*actual* inputs of the current analysis (FWI, fuel moisture, slope, land
cover, wind) with declared thresholds — never hardcoded per-location text.

Every factor reports:
    - the real value that was measured / modelled,
    - a qualitative level derived from declared thresholds,
    - its contribution to the score in points (where the score formula
      uses it directly),
    - whether it affects the score or only the spread modelling,
    - the provenance kind of its source.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..prediction.fwi import danger_class as fwi_danger_class

DISCLAIMER = (
    "This is a composite wildfire-risk indicator (0-100), not a probability "
    "of fire. It summarises current fire-weather danger, fuel dryness, "
    "terrain and land cover from real data."
)

SCORE_FORMULA_NOTE = (
    "Score = 100 * FWI / (FWI + 25), plus a slope contribution (up to +8), "
    "plus a fuel-moisture adjustment (+6 very dry / +3 dry / -4 moist), "
    "reduced to 30% where the dominant land cover is effectively "
    "non-burnable. Anchored to the real Canadian FWI of the day."
)


def _level(value: Optional[float], thresholds) -> Optional[str]:
    """Map a value to a level label via [(upper, label)] thresholds."""
    if value is None:
        return None
    for upper, label in thresholds:
        if value < upper:
            return label
    return thresholds[-1][1]


# Declared level scales (aligned with the score formula where it uses them).
_FWI_LEVELS = [(11.2, "Low"), (21.3, "Moderate"), (38.0, "High"), (float("inf"), "Extreme")]
_FMC_LEVELS = [(12.0, "Very dry"), (18.0, "Dry"), (30.0, "Normal"), (float("inf"), "Moist")]
_SLOPE_LEVELS = [(5.0, "Flat"), (12.0, "Gentle"), (20.0, "Steep"), (float("inf"), "Very steep")]
_WIND_LEVELS = [(12.0, "Calm"), (25.0, "Moderate"), (40.0, "Strong"), (float("inf"), "Very strong")]

# Level severity rank for UI colouring.
_LEVEL_RANK = {
    "Low": 0, "Calm": 0, "Flat": 0, "Moist": 0, "Non-burnable": 0,
    "Moderate": 1, "Gentle": 1, "Normal": 1, "Mixed": 1,
    "High": 2, "Dry": 2, "Steep": 2, "Strong": 2, "Burnable": 2,
    "Extreme": 3, "Very dry": 3, "Very steep": 3, "Very strong": 3,
}


def _fmc_adjustment(fmc: Optional[float]) -> Optional[float]:
    """Fuel-moisture adjustment exactly as used by the score formula."""
    if fmc is None:
        return None
    if fmc < 12.0:
        return 6.0
    if fmc < 18.0:
        return 3.0
    if fmc > 30.0:
        return -4.0
    return 0.0


def build_risk_explanation(
    *,
    fwi: Optional[float],
    fmc: Optional[float],
    slope: float,
    wind_kmh: float,
    landcover_label: Optional[str],
    burnable: bool,
    score: Optional[float],
    risk_class: Optional[str],
    fmc_source: Optional[str] = None,
) -> Dict:
    """
    Build the structured "Why this score?" block from real model inputs.

    All values are the ones actually used by the current analysis; factors
    whose inputs are unavailable are reported as unavailable.
    """
    factors = []

    # ---- Fire weather (FWI) -------------------------------------------
    fwi_level = _level(fwi, _FWI_LEVELS)
    factors.append({
        "key": "fire_weather",
        "label": "Fire weather (FWI)",
        "value": round(fwi, 1) if fwi is not None else None,
        "unit": "FWI",
        "level": fwi_level,
        "level_rank": _LEVEL_RANK.get(fwi_level),
        "contribution": (
            round(100.0 * fwi / (fwi + 25.0), 1) if fwi is not None else None
        ),
        "contribution_note": "base of the score (saturating in FWI)" if fwi is not None else None,
        "affects_score": True,
        "provenance_kind": "derived" if fwi is not None else "unavailable",
        "source": "Canadian FWI System from Open-Meteo daily data",
    })

    # ---- Fuel dryness (FMC) --------------------------------------------
    fmc_level = _level(fmc, _FMC_LEVELS)
    fmc_adj = _fmc_adjustment(fmc)
    factors.append({
        "key": "fuel_dryness",
        "label": "Fuel dryness",
        "value": round(fmc, 1) if fmc is not None else None,
        "unit": "% FMC",
        "level": fmc_level,
        "level_rank": _LEVEL_RANK.get(fmc_level),
        "contribution": fmc_adj,
        "contribution_note": (
            f"{'+' if (fmc_adj or 0) >= 0 else ''}{fmc_adj} points (dry fuel aggravates)"
            if fmc_adj else "no adjustment"
        ) if fmc is not None else None,
        "affects_score": True,
        "provenance_kind": "derived" if fmc is not None else "unavailable",
        "source": fmc_source or "Fuel-moisture estimate",
    })

    # ---- Terrain --------------------------------------------------------
    slope_level = _level(slope, _SLOPE_LEVELS)
    slope_contrib = round(min(slope, 45.0) / 45.0 * 8.0, 1)
    factors.append({
        "key": "terrain",
        "label": "Terrain (slope)",
        "value": round(slope, 1),
        "unit": "°",
        "level": slope_level,
        "level_rank": _LEVEL_RANK.get(slope_level),
        "contribution": slope_contrib,
        "contribution_note": f"+{slope_contrib} points (steeper slopes spread fire faster)",
        "affects_score": True,
        "provenance_kind": "observed",
        "source": "DEM (OpenTopoData; EU-DEM 25 m in Europe)",
    })

    # ---- Vegetation / land cover ---------------------------------------
    veg_level = "Burnable" if burnable else "Non-burnable"
    factors.append({
        "key": "vegetation",
        "label": "Vegetation / land cover",
        "value": landcover_label,
        "unit": None,
        "level": veg_level,
        "level_rank": _LEVEL_RANK.get(veg_level),
        "contribution": None if burnable else "score x 0.3",
        "contribution_note": (
            "dominant cover is burnable" if burnable
            else "dominant cover is effectively non-burnable; score reduced to 30%"
        ),
        "affects_score": True,
        "provenance_kind": "observed" if landcover_label else "unavailable",
        "source": "ESA WorldCover (10 m)",
    })

    # ---- Wind (context: spread modelling) -------------------------------
    wind_level = _level(wind_kmh, _WIND_LEVELS)
    factors.append({
        "key": "wind",
        "label": "Wind",
        "value": round(wind_kmh, 1),
        "unit": "km/h",
        "level": wind_level,
        "level_rank": _LEVEL_RANK.get(wind_level),
        "contribution": None,
        "contribution_note": "already reflected in the FWI; also drives the spread model",
        "affects_score": False,
        "provenance_kind": "modeled",
        "source": "Open-Meteo current weather (model grid)",
    })

    return {
        "score": score,
        "score_max": 100,
        "risk_class": risk_class,
        "fwi_danger_class": fwi_danger_class(fwi, simple=False) if fwi is not None else None,
        "factors": factors,
        "formula": SCORE_FORMULA_NOTE,
        "disclaimer": DISCLAIMER,
    }


def compact_factors(explanation: Dict) -> list:
    """Reduce an explanation to label/level pairs for compact UI display."""
    out = []
    for f in (explanation or {}).get("factors") or []:
        out.append({
            "key": f["key"],
            "label": f["label"],
            "value": f["value"],
            "unit": f["unit"],
            "level": f["level"],
            "level_rank": f["level_rank"],
        })
    return out
