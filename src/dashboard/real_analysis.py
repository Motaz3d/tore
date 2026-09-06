"""
Real-data analysis engine for Talaix.

Ties verified external data (geocoding, weather model, daily fire-weather
series, DEM, real Sentinel-2, ESA WorldCover, NASA FIRMS) to the Talaix
scientific models:

    real data -> FWI fire danger -> fuel moisture -> fire spread
              -> intervention comparison -> provenance-annotated report

Headline outputs:
    - Fire danger (Canadian FWI System, EFFIS classes) with 7-day trend
    - Composite wildfire risk score (0-100) and Low/Moderate/High/Extreme class
    - Rate of spread and screening spread-ellipse estimates
    - Baseline vs Talaix intervention comparison
    - Water-use efficiency ratio (WUER), evacuation safety margin

Every component carries a structured ``provenance`` entry:
    kind (observed | derived | modeled | forecast | unavailable),
    source, acquisition time, spatial/temporal resolution, quality flag and
    known limitations. When a value cannot be obtained it is reported as
    unavailable — never fabricated.
"""

from __future__ import annotations

import math
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np

from . import real_data
from .explain import build_risk_explanation
from .change import build_change_block
from .ecology import build_ecology_block
from .exposure import build_exposure_block
from .fire_evidence import build_fire_evidence
from .ignition import build_ignition_block
from .micro import build_micro_area_block
from .population import build_population_block
from .scenarios import build_scenarios
from .smoke import smoke_scenario
from .recommendations import build_recommendations, build_action_plan
from ..prediction.fire_spread import FireSpreadModel
from ..prediction.fuel_moisture import FuelMoistureModel
from ..prediction.fwi import compute_fwi_series, danger_class as fwi_danger_class
from ..hydration_control.water_optimiser import WaterOptimiser
from ..gis_mapping.copernicus_data import _estimate_fmc_from_ndmi


def _num(value, default=None):
    """Return a finite float if possible, else the default."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _prov(
    kind: str,
    source: str,
    acquired: Optional[str] = None,
    resolution: Optional[str] = None,
    temporal: Optional[str] = None,
    quality: str = "ok",
    limitations: Optional[str] = None,
) -> Dict:
    """Build a structured provenance record for one analysis component."""
    return {
        "kind": kind,  # observed | derived | modeled | forecast | unavailable
        "source": source,
        "acquired": acquired,
        "resolution": resolution,
        "temporal": temporal,
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
        "quality": quality,
        "limitations": limitations,
    }


class TalaixRealAnalyser:
    """
    Run a full Talaix analysis for a location using real data.

    Usage::

        analyser = TalaixRealAnalyser()
        result = analyser.analyse("Clervaux, Luxembourg")
        result = analyser.analyse_point(49.9, 6.03, name="Clervaux")
    """

    RISK_CLASSES = [(25.0, "Low"), (45.0, "Moderate"), (65.0, "High"), (101.0, "Extreme")]

    #: Public analysis stages (id, label, source) — the progressive pipeline.
    STAGES = [
        ("location", "Location identified", "Coordinates / Nominatim"),
        ("weather", "Weather conditions", "Open-Meteo"),
        ("fire_danger", "Fire weather (FWI)", "Canadian FWI System"),
        ("terrain", "Terrain", "EU-DEM / SRTM (OpenTopoData)"),
        ("satellite", "Satellite observation", "Copernicus Sentinel-2"),
        ("fuel", "Fuel moisture", "Sentinel-2 NDMI + soil moisture"),
        ("landcover", "Vegetation & fuel model", "ESA WorldCover"),
        ("fires", "Active fire observations", "NASA FIRMS"),
        ("risk", "Risk calculation", "Talaix models"),
        ("context", "Context & exposure", "OpenStreetMap + scene grid"),
        ("solutions", "Solutions & recommendations", "Talaix engines"),
        ("assembly", "Final assembly", "Talaix"),
    ]

    def __init__(
        self,
        fuel_model: str = "TL3",
        intervention_fmc_increase: float = 20.0,
        application_rate_m3_per_h: float = 50.0,
        conventional_water_m3: float = 1500.0,
        intervention_water_m3: float = 450.0,
        use_landcover_fuel: bool = True,
    ) -> None:
        self.fuel_model = fuel_model
        self.intervention_fmc_increase = intervention_fmc_increase
        self.application_rate_m3_per_h = application_rate_m3_per_h
        self.conventional_water_m3 = conventional_water_m3
        self.intervention_water_m3 = intervention_water_m3
        self.use_landcover_fuel = use_landcover_fuel

        self.moisture_model = FuelMoistureModel()
        self.water_optimiser = WaterOptimiser()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def analyse(self, query: str) -> Dict:
        """Run the full pipeline for a free-text location string."""
        geo = real_data.geocode_location(query)
        if "error" in geo:
            return {"error": geo["error"], "query": query}
        result = self.analyse_point(geo["lat"], geo["lon"], name=geo["name"])
        result["query"] = query
        result["location"]["source"] = geo["source"]
        result["provenance"]["geocoding"] = _prov(
            "observed", geo["source"], resolution="point", quality="ok"
        )
        return result

    def analyse_point(
        self,
        lat: float,
        lon: float,
        name: Optional[str] = None,
        on_stage=None,
    ) -> Dict:
        """
        Run the full pipeline for coordinates.

        ``on_stage(stage_id, status, detail)`` is an optional callback fired
        at every honest stage transition (running / complete / unavailable)
        with small real detail values — this powers the progressive
        analysis UX and the job API. The returned dict is identical whether
        or not a callback is supplied.
        """
        def _mark(stage_id: str, status: str, detail: Optional[Dict] = None) -> None:
            if on_stage is not None:
                try:
                    on_stage(stage_id, status, detail or {})
                except Exception:
                    pass  # progress reporting must never break the analysis

        lat, lon = float(lat), float(lon)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"error": "Coordinates out of range"}

        provenance: Dict[str, Dict] = {}

        # ---- Stage: location -------------------------------------------
        _mark("location", "running")
        location = {"name": name or f"{lat:.4f}, {lon:.4f}",
                    "latitude": lat, "longitude": lon}
        _mark("location", "complete", {"name": location["name"],
                                       "latitude": lat, "longitude": lon})

        # ---- Stage: weather --------------------------------------------
        _mark("weather", "running")
        weather = real_data.fetch_weather_current(lat, lon)
        daily = real_data.fetch_daily_fire_weather(lat, lon)
        provenance["weather"] = (
            _prov("modeled", weather.get("source", "Open-Meteo"), acquired=weather.get("timestamp"),
                  resolution="~11 km (model grid)", temporal="current",
                  limitations="Numerical weather model output, not a station observation.")
            if "error" not in weather
            else _prov("unavailable", "Open-Meteo", quality="missing", limitations=weather.get("error"))
        )
        if "error" not in weather:
            _mark("weather", "complete", {
                "temperature_c": weather.get("temperature_c"),
                "wind_kmh": weather.get("wind_speed_kmh"),
                "humidity_pct": weather.get("relative_humidity_pct"),
                "source": weather.get("source"),
            })
        else:
            _mark("weather", "unavailable", {"reason": weather.get("error")})

        # ---- Stage: fire danger (FWI) ----------------------------------
        _mark("fire_danger", "running")
        fwi_block = self._compute_fire_danger(daily, provenance)
        if fwi_block.get("available"):
            _mark("fire_danger", "complete", {
                "fwi": fwi_block.get("fwi"), "class": fwi_block.get("class"),
                "date": fwi_block.get("date"),
                "source": "Canadian FWI System (Open-Meteo daily data)",
            })
        else:
            _mark("fire_danger", "unavailable", {
                "reason": (provenance.get("fire_danger") or {}).get("limitations")
                or "FWI series unavailable"})

        # ---- Stage: terrain --------------------------------------------
        _mark("terrain", "running")
        terrain = real_data.fetch_terrain(lat, lon)
        provenance["terrain"] = (
            _prov("observed", terrain.get("source", "DEM"), resolution=terrain.get("resolution"))
            if "error" not in terrain
            else _prov("unavailable", "DEM (OpenTopoData)", quality="missing",
                       limitations=terrain.get("error"))
        )
        if "error" not in terrain:
            _mark("terrain", "complete", {
                "elevation_m": terrain.get("elevation_m"),
                "slope_degrees": terrain.get("slope_degrees"),
                "source": terrain.get("source"),
            })
        else:
            _mark("terrain", "unavailable", {"reason": terrain.get("error")})

        # ---- Stage: satellite ------------------------------------------
        _mark("satellite", "running")
        satellite = real_data.fetch_satellite_data(lat, lon)
        provenance["satellite"] = (
            _prov("observed", satellite.get("source", "Sentinel-2"),
                  acquired=(satellite.get("observation_date") or "")[:10],
                  resolution="10 m",
                  limitations="Optical sensor: unavailable under persistent cloud cover.")
            if "error" not in satellite
            else _prov("unavailable", satellite.get("source", "Sentinel-2"), quality="missing",
                       limitations=satellite.get("error"))
        )
        if "error" not in satellite:
            _mark("satellite", "complete", {
                "observation_date": (satellite.get("observation_date") or "")[:10],
                "ndvi": satellite.get("ndvi"), "ndmi": satellite.get("ndmi"),
                "source": satellite.get("source"),
            })
        else:
            _mark("satellite", "unavailable", {"reason": satellite.get("error")})

        # ---- Stage: fuel moisture --------------------------------------
        _mark("fuel", "running")
        fmc_baseline, fmc_source, fmc_prov = self._derive_fmc(weather, satellite)
        provenance["fuel_moisture"] = fmc_prov
        if fmc_baseline is not None:
            _mark("fuel", "complete", {"fmc_pct": fmc_baseline, "source": fmc_source})
        else:
            _mark("fuel", "unavailable", {"reason": fmc_source})

        # ---- Stage: land cover / fuel model ----------------------------
        _mark("landcover", "running")
        landcover = self._fetch_landcover(lat, lon)
        fuel_model = self.fuel_model
        if self.use_landcover_fuel and "error" not in landcover:
            fuel_model = landcover.get("fuel_model") or self.fuel_model
        spread_model = FireSpreadModel(fuel_model=fuel_model)
        provenance["landcover"] = (
            _prov("observed", landcover.get("source", "ESA WorldCover"),
                  resolution=landcover.get("resolution"),
                  limitations="Fuel-model assignment is a screening approximation.")
            if "error" not in landcover
            else _prov("unavailable", "ESA WorldCover", quality="missing",
                       limitations=landcover.get("error"))
        )
        if "error" not in landcover:
            _mark("landcover", "complete", {
                "dominant_label": landcover.get("dominant_label"),
                "fuel_model": fuel_model, "source": landcover.get("source"),
            })
        else:
            _mark("landcover", "unavailable", {"reason": landcover.get("error")})

        # ---- Stage: active fires ---------------------------------------
        _mark("fires", "running")
        fires = real_data.fetch_active_fires(lat, lon)
        provenance["active_fires"] = (
            _prov("observed", fires.get("source", "NASA FIRMS"),
                  resolution=fires.get("resolution"), temporal=f"{fires.get('days', 5)} days")
            if fires.get("available")
            else _prov("unavailable", "NASA FIRMS", quality="missing",
                       limitations=fires.get("error"))
        )
        if fires.get("available"):
            _mark("fires", "complete", {
                "count": fires.get("count", 0), "days": fires.get("days"),
                "source": fires.get("source"),
            })
        else:
            _mark("fires", "unavailable", {"reason": fires.get("error")})

        # ---- Stage: risk calculation -----------------------------------
        _mark("risk", "running")
        wind_kmh = _num(weather.get("wind_speed_kmh"), default=0.0) or 0.0
        wind_dir = _num(weather.get("wind_direction_deg"), default=0.0) or 0.0
        slope = _num(terrain.get("slope_degrees"), default=0.0) or 0.0
        aspect = _num(terrain.get("aspect_degrees"), default=0.0) or 0.0
        wind_u, wind_v = self._wind_vector(wind_kmh, wind_dir)
        fmc_for_spread = fmc_baseline if fmc_baseline is not None else 8.0

        baseline = spread_model.compute_ros(
            fmc=fmc_for_spread, wind_speed_kmh=wind_kmh, slope_degrees=slope,
            wind_u=wind_u, wind_v=wind_v, wind_direction=wind_dir, aspect=aspect,
        )
        spread_probability = spread_model.probability_of_spread(
            fmc=fmc_for_spread, wind_speed_kmh=wind_kmh, slope_degrees=slope
        )

        intervention = None
        target_fmc = None
        if fmc_baseline is not None:
            target_fmc = _clamp(fmc_baseline + self.intervention_fmc_increase, 0.0, 100.0)
            intervention = spread_model.compute_ros(
                fmc=target_fmc, wind_speed_kmh=wind_kmh, slope_degrees=slope,
                wind_u=wind_u, wind_v=wind_v, wind_direction=wind_dir, aspect=aspect,
            )

        burnable = landcover.get("burnable", True) if "error" not in landcover else True
        risk_baseline = self._risk_score(
            fwi=fwi_block.get("fwi"), slope=slope, fmc=fmc_baseline,
            wind_kmh=wind_kmh, burnable=burnable,
        )
        risk_intervention = (
            self._risk_score(
                fwi=fwi_block.get("fwi"), slope=slope, fmc=target_fmc,
                wind_kmh=wind_kmh, burnable=burnable,
            )
            if target_fmc is not None
            else None
        )
        risk_class = self._risk_class(risk_baseline)

        provenance["risk_score"] = _prov(
            "derived", "Talaix composite (FWI + fuel moisture + slope + land cover)",
            limitations="Screening-level score, not a validated local fire-danger rating.",
        )
        provenance["fire_spread"] = _prov(
            "modeled", f"Talaix FireSpreadModel (fuel {fuel_model})",
            limitations="Simplified ROS model; ellipse is a screening estimate without "
                        "spotting, fuel breaks or fire-suppression effects.",
        )

        wuer = None
        if risk_baseline is not None and risk_intervention is not None:
            wuer = self.water_optimiser.compute_wuer(
                risk_baseline=risk_baseline,
                risk_hydrashield=risk_intervention,
                water_volume_m3=self.intervention_water_m3,
            )

        arrival_baseline = _num(baseline.ros_reduced, default=0.0) or 0.0
        esm_baseline = self._evacuation_margin(arrival_baseline)
        arrival_intervention = (
            _num(intervention.ros_reduced, default=arrival_baseline)
            if intervention is not None
            else arrival_baseline
        )
        esm_intervention = self._evacuation_margin(arrival_intervention)

        spread_ellipse = self._spread_ellipses(
            ros_m_min=arrival_baseline, wind_speed_kmh=wind_kmh, wind_direction_deg=wind_dir
        )
        spread_ellipse_intervention = self._spread_ellipses(
            ros_m_min=arrival_intervention, wind_speed_kmh=wind_kmh, wind_direction_deg=wind_dir
        )

        trend = self._fwi_trend(fwi_block.get("series") or [])

        landcover_label = (
            landcover.get("dominant_label") if "error" not in landcover else None
        )
        explanation = build_risk_explanation(
            fwi=fwi_block.get("fwi") if fwi_block.get("available") else None,
            fmc=fmc_baseline,
            slope=slope,
            wind_kmh=wind_kmh,
            landcover_label=landcover_label,
            burnable=burnable,
            score=risk_baseline,
            risk_class=risk_class,
            fmc_source=fmc_source,
        )
        provenance["risk_explanation"] = _prov(
            "derived", "Talaix risk-score decomposition (declared thresholds)",
            limitations="Levels are qualitative summaries of the real inputs, "
                        "not additional measurements.",
        )
        if risk_baseline is not None:
            _mark("risk", "complete", {"risk": risk_baseline, "risk_class": risk_class})
        else:
            _mark("risk", "unavailable", {
                "reason": "Insufficient real inputs for a risk score — none generated."})

        result = {
            "location": location,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "terrain": terrain,
            "weather": weather,
            "satellite": satellite,
            "landcover": landcover,
            "active_fires": fires,
            "fire_danger": fwi_block,
            "fire_danger_trend": trend,
            "risk_explanation": explanation,
            "analysis": {
                "fuel_moisture_baseline_pct": fmc_baseline,
                "fuel_moisture_source": fmc_source,
                "fuel_moisture_intervention_pct": target_fmc,
                "mefmi_pct": (
                    round(target_fmc - fmc_baseline, 2)
                    if (target_fmc is not None and fmc_baseline is not None)
                    else None
                ),
                "probability_of_spread": round(spread_probability, 4),
                "fire_spread": {
                    "fuel_model": fuel_model,
                    "ros_baseline_m_min": _num(baseline.ros_baseline),
                    "ros_current_m_min": _num(baseline.ros_reduced),
                    "ros_intervention_m_min": (
                        _num(intervention.ros_reduced) if intervention is not None else None
                    ),
                    "ros_horizontal_m_min": _num(baseline.ros_horizontal),
                    "ros_crown_m_min": _num(baseline.ros_crown),
                    "spread_ellipse": spread_ellipse,
                    "spread_ellipse_intervention": spread_ellipse_intervention,
                },
                "risk": {
                    "baseline": risk_baseline,
                    "class": risk_class,
                    "intervention": risk_intervention,
                    "intervention_class": (
                        self._risk_class(risk_intervention) if risk_intervention is not None else None
                    ),
                    "reduction_percent": (
                        round((risk_baseline - risk_intervention) / risk_baseline * 100.0, 1)
                        if (risk_baseline and risk_intervention is not None)
                        else None
                    ),
                },
                "evacuation_safety_margin_min": {
                    "baseline": esm_baseline,
                    "intervention": esm_intervention,
                    "improvement_min": round(esm_intervention - esm_baseline, 1),
                },
                "wuer": wuer.to_dict() if wuer is not None else None,
                "water_savings_pct": round(
                    self.water_optimiser.water_savings(
                        self.conventional_water_m3, self.intervention_water_m3
                    ),
                    1,
                ),
            },
            "provenance": provenance,
            # Backward-compatible summary for the Dash dashboard.
            "data_quality": self._data_quality(weather, terrain, satellite),
            "methodology": {
                "fuel_model": fuel_model,
                "intervention_fmc_increase_pct": self.intervention_fmc_increase,
                "intervention_water_m3": self.intervention_water_m3,
                "conventional_water_m3": self.conventional_water_m3,
                "note": "Risk score is anchored to the Canadian FWI computed from real "
                        "daily weather; fuel moisture uses real Sentinel-2 NDMI when a "
                        "cloud-free scene exists, otherwise real soil-moisture model "
                        "output. FMC calibration coefficients are placeholders pending "
                        "Phase-6 fitting against measured data.",
            },
        }

        # ---- Stage: context (change / climate / exposure / micro) ------
        _mark("context", "running")
        result["change"] = build_change_block(daily, fwi_block, slope, satellite)
        provenance["change"] = _prov(
            "derived", "FWI + Open-Meteo daily series (real)",
            limitations=result["change"].get("basis_note") or
            result["change"].get("reason"),
        )

        result["climate"] = self._climate_summary(daily)
        provenance["climate"] = (
            _prov("derived", "Open-Meteo daily aggregates (real)",
                  limitations=result["climate"].get("note"))
            if result["climate"].get("available")
            else _prov("unavailable", "Open-Meteo daily series", quality="missing")
        )

        # ---- Multi-source fire evidence (transparent, never merged) ----
        result["fire_evidence"] = build_fire_evidence(lat, lon)
        provenance["fire_evidence"] = _prov(
            (result["fire_evidence"].get("provenance") or {}).get("kind", "unavailable"),
            (result["fire_evidence"].get("provenance") or {}).get(
                "source", "NASA FIRMS"),
            quality=(result["fire_evidence"].get("provenance") or {}).get("quality", "ok"),
            limitations=(result["fire_evidence"].get("provenance") or {}).get("limitations"),
        )

        result["exposure"] = build_exposure_block(result)
        provenance["exposure"] = _prov(
            (result["exposure"].get("provenance") or {}).get("kind", "unavailable"),
            (result["exposure"].get("provenance") or {}).get(
                "source", "OpenStreetMap (ohsome / Overpass)"),
            quality=(result["exposure"].get("provenance") or {}).get("quality", "ok"),
            limitations=(result["exposure"].get("provenance") or {}).get("limitations"),
        )

        result["micro_area"] = build_micro_area_block(result)
        provenance["micro_area"] = _prov(
            "derived", "Measured Sentinel-2 scene grid + declared layer resolutions",
            limitations=(result["micro_area"].get("provenance") or {}).get("limitations"),
        )

        # ---- Population exposure + ignition + smoke (kept separate) ----
        result["population"] = build_population_block(result)
        provenance["population"] = _prov(
            (result["population"].get("provenance") or {}).get("kind", "unavailable"),
            (result["population"].get("provenance") or {}).get(
                "source", "WorldPop gridded population"),
            quality=(result["population"].get("provenance") or {}).get("quality", "ok"),
            limitations=(result["population"].get("provenance") or {}).get("limitations"),
        )

        result["ignition"] = build_ignition_block(result)
        provenance["ignition"] = _prov(
            (result["ignition"].get("provenance") or {}).get("kind", "unavailable"),
            (result["ignition"].get("provenance") or {}).get(
                "source", "Talaix ignition layer"),
            quality=(result["ignition"].get("provenance") or {}).get("quality", "ok"),
            limitations=(result["ignition"].get("provenance") or {}).get("limitations"),
        )

        # Smoke Intelligence: only the SCENARIO mode belongs in the analysis
        # (no fire is claimed to exist). Observed-fire smoke is served by
        # /api/smoke with its own FIRMS-backed provenance.
        try:
            result["smoke_scenario"] = smoke_scenario(float(lat), float(lon))
        except Exception as exc:
            result["smoke_scenario"] = {"error": str(exc), "mode": "scenario"}
        provenance["smoke_scenario"] = _prov(
            (result["smoke_scenario"].get("provenance") or {}).get("kind", "unavailable"),
            (result["smoke_scenario"].get("provenance") or {}).get(
                "source", "Open-Meteo wind profile (scenario transport)"),
            quality=(result["smoke_scenario"].get("provenance") or {}).get("quality", "ok"),
            limitations=(result["smoke_scenario"].get("provenance") or {}).get("limitations")
            or result["smoke_scenario"].get("error"),
        )
        context_unavailable = []
        if not result["change"].get("available"):
            context_unavailable.append("temporal comparison")
        if result["exposure"].get("status") != "ok":
            context_unavailable.append("OSM exposure")
        if context_unavailable:
            _mark("context", "unavailable" if len(context_unavailable) == 2 else "complete",
                  {"partial": context_unavailable})
        else:
            _mark("context", "complete", {})

        # ---- Stage: solutions ------------------------------------------
        _mark("solutions", "running")
        result["ecology"] = build_ecology_block(result)
        provenance["ecology"] = _prov(
            "derived", (result["ecology"].get("provenance") or {}).get(
                "source", "Talaix ecology engine"),
            quality=(result["ecology"].get("provenance") or {}).get("quality", "ok"),
            limitations=(result["ecology"].get("provenance") or {}).get("limitations"),
        )

        result["scenarios"] = build_scenarios(result)
        provenance["scenarios"] = _prov(
            "modeled", "Talaix FireSpreadModel + composite risk score",
            limitations="Screening-level scenario estimates; effects beyond the "
                        "models are reported as not quantified.",
        )

        result["recommendations"] = build_recommendations(result)
        result["action_plan"] = build_action_plan(result, result["recommendations"])
        provenance["recommendations"] = _prov(
            "derived", "Talaix evidence-linked rule engine",
            limitations="Recommendations are generated from the detected "
                        "conditions; they do not guarantee prevention.",
        )
        _mark("solutions", "complete", {
            "recommendations": len(result["recommendations"]),
            "ecology_status": result["ecology"].get("status"),
        })

        # ---- Stage: assembly --------------------------------------------
        _mark("assembly", "running")
        result["provenance"] = provenance
        _mark("assembly", "complete", {})
        return result

    # ------------------------------------------------------------------
    # Fire danger (FWI)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_fire_danger(daily: Dict, provenance: Dict) -> Dict:
        """Run the FWI System over the real daily fire-weather series."""
        if "error" in daily or not daily.get("days"):
            provenance["fire_danger"] = _prov(
                "unavailable", "Open-Meteo daily series", quality="missing",
                limitations=daily.get("error", "No daily series"),
            )
            return {"available": False}

        # Build FWI inputs from daily aggregates (screening approximation).
        series_in = []
        for d in daily["days"]:
            temp = _num(d.get("temp_max_c"))
            rh = _num(d.get("rh_min_pct"), default=_num(d.get("rh_mean_pct")))
            wind = _num(d.get("wind_mean_kmh"), default=_num(d.get("wind_max_kmh")))
            rain = _num(d.get("precipitation_mm"), default=0.0)
            if temp is None or rh is None or wind is None:
                continue
            series_in.append(
                {"date": d["date"], "temp_c": temp, "rh_pct": rh,
                 "wind_kmh": wind, "rain_mm": rain or 0.0}
            )

        if len(series_in) < 5:
            provenance["fire_danger"] = _prov(
                "unavailable", "Open-Meteo daily series", quality="missing",
                limitations="Daily series too short to spin up the FWI System.",
            )
            return {"available": False}

        fwi_days = compute_fwi_series(series_in)
        today_str = date.today().isoformat()
        past = [d for d in fwi_days if d.date <= today_str]
        forecast = [d for d in fwi_days if d.date > today_str]
        current = past[-1] if past else fwi_days[0]

        provenance["fire_danger"] = _prov(
            "derived", "Canadian FWI System (Van Wagner 1987) from Open-Meteo daily data",
            acquired=current.date, temporal="daily",
            limitations=daily.get("note"),
        )

        return {
            "available": True,
            "fwi": round(current.fwi, 1),
            "class": current.danger_class,
            "effis_class": fwi_danger_class(current.fwi, simple=False),
            "ffmc": round(current.ffmc, 1),
            "dmc": round(current.dmc, 1),
            "dc": round(current.dc, 1),
            "isi": round(current.isi, 1),
            "bui": round(current.bui, 1),
            "dsr": round(current.dsr, 2),
            "date": current.date,
            "series": [d.to_dict() for d in past[-14:]],
            "forecast": [d.to_dict() for d in forecast[:7]],
        }

    # ------------------------------------------------------------------
    # Fuel moisture
    # ------------------------------------------------------------------
    def _derive_fmc(self, weather: Dict, satellite: Dict):
        """
        Best-available FMC estimate:
        1. Real Sentinel-2 NDMI (when a usable scene exists), blended 60/40
           with the soil-moisture estimate.
        2. Capillary transfer from real surface soil moisture.
        3. Relative-humidity equilibrium proxy.
        Returns (fmc, source_label, provenance).
        """
        fmc_wx = None
        sm = weather.get("soil_moisture_m3m3")
        if sm is not None:
            sm_c = _clamp(float(sm), 0.0, 1.0)
            fmc_wx = round(100.0 * 0.35 * min(sm_c / 0.45, 1.0), 2)

        if "error" not in satellite and satellite.get("ndmi") is not None:
            ndmi_value = float(satellite["ndmi"])
            sat_fmc = float(_estimate_fmc_from_ndmi(np.array([[ndmi_value]]))[0][0])
            if fmc_wx is not None:
                fmc = round(0.4 * fmc_wx + 0.6 * sat_fmc, 2)
                src = "Derived (60% real Sentinel-2 NDMI + 40% soil-moisture model)"
            else:
                fmc = round(sat_fmc, 2)
                src = "Derived (real Sentinel-2 NDMI)"
            return fmc, src, _prov(
                "derived", "Sentinel-2 NDMI (real) + Open-Meteo soil moisture",
                acquired=(satellite.get("observation_date") or "")[:10],
                resolution="10 m scene average",
                limitations="NDMI->FMC calibration is a placeholder pending field fitting.",
            )

        if fmc_wx is not None:
            return fmc_wx, "Derived (capillary transfer from real soil moisture)", _prov(
                "derived", "Open-Meteo soil_moisture_0_to_7cm",
                acquired=weather.get("timestamp"),
                limitations="Modelled soil moisture; capillary-transfer coefficient k=0.35 "
                            "is a literature value, not locally calibrated.",
            )

        rh = _num(weather.get("relative_humidity_pct"))
        if rh is not None:
            fmc = round(3.0 + 22.0 * (_clamp(rh, 0.0, 100.0) / 100.0) ** 1.5, 2)
            return fmc, "Derived (relative-humidity equilibrium proxy)", _prov(
                "derived", "Open-Meteo relative humidity",
                acquired=weather.get("timestamp"),
                limitations="Coarse 1-h dead-fuel equilibrium proxy.",
            )

        return None, "Unavailable (no soil-moisture or satellite observation)", _prov(
            "unavailable", "—", quality="missing",
            limitations="Neither satellite nor weather-moisture inputs available.",
        )

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------
    @classmethod
    def _risk_class(cls, score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        for threshold, label in cls.RISK_CLASSES:
            if score < threshold:
                return label
        return "Extreme"

    @staticmethod
    def _risk_score(
        fwi: Optional[float],
        slope: float,
        fmc: Optional[float],
        wind_kmh: float,
        burnable: bool = True,
    ) -> Optional[float]:
        """
        Composite screening risk score 0-100, anchored to the real FWI.

            base = 100 * FWI / (FWI + 25)      (saturating in FWI)
            + slope contribution (up to +8)
            + fuel-moisture adjustment (dry fuel aggravates, moist moderates)
            x 0.3 when the dominant land cover is effectively non-burnable

        Falls back to the wind/humidity formulation when no FWI is available.
        """
        if fwi is None:
            if fmc is None:
                return None
            risk = 50.0 + (wind_kmh / 50.0) * 30.0 + min(slope, 45.0) / 45.0 * 20.0 - fmc - 15.0
        else:
            risk = 100.0 * fwi / (fwi + 25.0)
            risk += min(slope, 45.0) / 45.0 * 8.0
            if fmc is not None:
                if fmc < 12.0:
                    risk += 6.0
                elif fmc < 18.0:
                    risk += 3.0
                elif fmc > 30.0:
                    risk -= 4.0
        if not burnable:
            risk *= 0.3
        return round(_clamp(risk, 0.0, 100.0), 1)

    # ------------------------------------------------------------------
    # Spread ellipses (screening estimate)
    # ------------------------------------------------------------------
    @staticmethod
    def _spread_ellipses(ros_m_min: float, wind_speed_kmh: float, wind_direction_deg: float) -> Dict:
        """
        Simple elliptical spread estimate for 1/3/6 h horizons.

        Length-to-breadth ratio grows with wind speed (documented heuristic,
        LB = clamp(1 + 0.025 * wind_kmh, 1, 4)). Heading is the direction the
        wind blows TO. This is a screening approximation, not a spatial
        simulator.
        """
        if ros_m_min <= 0:
            return {"available": False}
        heading = (wind_direction_deg + 180.0) % 360.0
        lb = _clamp(1.0 + 0.025 * wind_speed_kmh, 1.0, 4.0)
        horizons = {}
        for hours in (1, 3, 6):
            a = ros_m_min * 60.0 * hours  # semi-major axis (m), downwind
            b = a / lb
            horizons[f"{hours}h"] = {
                "downwind_distance_m": round(a, 0),
                "max_width_m": round(2 * b, 0),
                "area_km2": round(math.pi * a * b / 1e6, 3),
            }
        return {
            "available": True,
            "heading_deg": round(heading, 1),
            "length_to_breadth": round(lb, 2),
            "horizons": horizons,
            "model": "Elliptical screening estimate (no spotting/heterogeneity)",
        }

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _climate_summary(daily: Dict) -> Dict:
        """
        Recent-climate summary from the real daily aggregates (past days only).

        Used by the ecology layer as a screening climate signal. Declared
        approximation: the window is the available past series (~3 weeks),
        not a climatological normal.
        """
        days = [d for d in (daily.get("days") or []) if d.get("date")]
        from datetime import date as _date

        today = _date.today().isoformat()
        past = [d for d in days if d["date"] <= today]
        if not past:
            return {"available": False}
        tmax = [d.get("temp_max_c") for d in past if d.get("temp_max_c") is not None]
        rain = [d.get("precipitation_mm") or 0.0 for d in past]
        rh_min = [d.get("rh_min_pct") for d in past if d.get("rh_min_pct") is not None]
        return {
            "available": True,
            "window_days": len(past),
            "mean_temp_max_c": round(sum(tmax) / len(tmax), 1) if tmax else None,
            "total_precip_mm": round(sum(rain), 1),
            "mean_rh_min_pct": round(sum(rh_min) / len(rh_min), 1) if rh_min else None,
            "note": "Screening climate signal from the recent real daily series; "
                    "not a climatological normal.",
        }

    @staticmethod
    def _fwi_trend(series: List[Dict]) -> Dict:
        """Classify the recent FWI trend as rising / steady / falling."""
        vals = [d.get("fwi") for d in series if d.get("fwi") is not None]
        if len(vals) < 5:
            return {"trend": "unknown", "slope_per_day": None}
        x = np.arange(len(vals), dtype=float)
        slope = float(np.polyfit(x, np.asarray(vals, dtype=float), 1)[0])
        label = "steady"
        if slope > 0.5:
            label = "rising"
        elif slope < -0.5:
            label = "falling"
        return {"trend": label, "slope_per_day": round(slope, 2), "window_days": len(vals)}

    @staticmethod
    def _wind_vector(speed_kmh: float, direction_deg: float):
        rad = math.radians(direction_deg)
        u = -speed_kmh * math.sin(rad)
        v = -speed_kmh * math.cos(rad)
        return u, v

    @staticmethod
    def _evacuation_margin(ros_m_min: float) -> float:
        """
        Evacuation safety margin for a 1 km reference front.

        Assumes a nominal 120-minute evacuation window, 15-minute operational
        margin and 10-minute uncertainty buffer (declared assumptions).
        """
        if ros_m_min <= 0:
            return 9999.0
        fire_arrival = 1000.0 / ros_m_min
        return round(120.0 - fire_arrival - 15.0 - 10.0, 1)

    def _fetch_landcover(self, lat: float, lon: float) -> Dict:
        if not self.use_landcover_fuel:
            return {"error": "Land-cover lookup disabled"}
        try:
            from ..gis_mapping.landcover import fetch_landcover

            return fetch_landcover(lat, lon)
        except Exception as exc:
            return {"error": f"Land-cover lookup failed: {exc}"}

    @staticmethod
    def _data_quality(weather: Dict, terrain: Dict, satellite: Dict) -> Dict:
        """Backward-compatible availability summary (used by the Dash app)."""
        components = {
            "geocoding": "OBSERVED (OpenStreetMap)",
            "weather": "MODEL (Open-Meteo)",
            "terrain": "DEM (OpenTopoData)",
            "satellite_ndvi_ndmi": (
                "OBSERVED (Sentinel-2)" if "error" not in satellite else "UNAVAILABLE"
            ),
            "fire_history_effis": "NOT YET INTEGRATED",
        }
        availability = {
            "weather_current": weather.get("temperature_c") is not None,
            "soil_moisture": weather.get("soil_moisture_m3m3") is not None,
            "terrain": "elevation_m" in terrain,
            "satellite": "error" not in satellite,
        }
        return {"components": components, "availability": availability}
