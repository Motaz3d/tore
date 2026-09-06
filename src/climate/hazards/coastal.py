"""
Coastal / sea-exposure hazard plugin (Stage 4 foundation).

Real-data analyses only:

- **Wave conditions** — Open-Meteo Marine API (ECMWF WAM): daily wave-height
  and wave-period maxima. Dates up to the analysis time are a model nowcast
  (temporal OBSERVED window), later dates are labelled FORECAST.
- **Low-elevation screening** — DEM elevation at the point (OpenTopoData),
  with declared screening bands. Screening only — NOT a flood or surge model.
- **Exposure** — mapped OSM buildings/roads/critical facilities near the
  point.
- **Sea-level rise** — ONLY as a structurally separated ``projections``
  block labelled PROJECTED/SCENARIO, citing IPCC AR6 (approximate published
  likely global-mean ranges for 2100 vs 1995–2014). Never mixed into
  observations.

Explicitly NOT claimed: storm-surge modelling, erosion rates, flood extents.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from ..ontology import ClaimStatus, Confidence, EvidenceClass, TemporalClass
from ..evidence import EvidenceRecord, content_hash
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec
from . import _series

_HISTORY_DAYS = 365
_FORECAST_DAYS = 7
_OSM_RADIUS_M = 2000

# IPCC AR6 WG1 Summary for Policymakers (2021), Table SPM.1 — approximate
# published "likely" global-mean sea-level rise by 2100 relative to
# 1995–2014. Stored as sourced constants; SCENARIO-labelled; global means.
_SLR_PROJECTIONS = [
    {"scenario": "SSP1-2.6", "likely_range_2100_m": [0.28, 0.55]},
    {"scenario": "SSP2-4.5", "likely_range_2100_m": [0.44, 0.76]},
    {"scenario": "SSP5-8.5", "likely_range_2100_m": [0.63, 1.01]},
]
_SLR_SOURCE = (
    "IPCC AR6 WG1 Summary for Policymakers (2021), Table SPM.1 — approximate "
    "published 'likely' global-mean ranges by 2100 relative to 1995–2014"
)

# Declared low-elevation screening bands (metres above sea level).
_ELEV_BANDS = ((2.0, "very low-lying"), (5.0, "low-lying"), (10.0, "moderately low"))


class CoastalModule(HazardModule):
    id = "coastal"
    name = "Coastal / sea exposure"
    tagline = "Wave conditions, low-elevation screening, coastal exposure, labelled sea-level-rise scenarios."

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "Ocean waves (Open-Meteo Marine API, ECMWF WAM)": {
                "start": "2022 (per Open-Meteo documentation)",
                "end": "nowcast + short forecast",
            },
            "Sea-level rise (IPCC AR6 scenarios)": {
                "start": "1995–2014 reference",
                "end": "2100 projections",
            },
        }

    # -- analysis ------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard import real_data as rd
        from ...dashboard import exposure as osm

        today = date.today()
        start = (today - timedelta(days=_HISTORY_DAYS)).isoformat()
        end = (today + timedelta(days=_FORECAST_DAYS)).isoformat()
        location = {"lat": lat, "lon": lon}

        marine = rd.fetch_marine(lat, lon, start, end)
        elevation = rd.fetch_elevation(lat, lon)
        osm_ctx = osm.fetch_osm_context(lat, lon, _OSM_RADIUS_M)

        blocks: Dict[str, Any] = {}
        evidence: List[Dict[str, Any]] = []
        provenance: Dict[str, Any] = {}
        data_ok = 0

        # -- waves --------------------------------------------------------
        wave_block, wave_pct = self._waves_block(marine, today)
        blocks["waves"] = wave_block
        if wave_block.get("status") == "ok":
            data_ok += 1
            rec = EvidenceRecord(
                EvidenceClass.OPEN_DATA_OFFICIAL.value,
                ClaimStatus.MODELLED.value,
                TemporalClass.OBSERVED.value,
                marine["source"],
                dataset="ECMWF WAM daily wave height/period maxima",
                provider_url="https://open-meteo.com/en/docs/marine-weather-api",
                link=marine.get("request_url"),
                location={"lat": lat, "lon": lon},
                reference_period={"start": start, "end": end},
                method=(
                    "Latest nowcast wave-height max percentile vs the past year's daily "
                    "series; dates after the analysis time are FORECAST and labelled so."
                ),
                resolution="wave-model grid (tens of km)",
                limitations="Wave-model output; coastal bathymetry and surf zone not resolved.",
                content_hash=content_hash(
                    {"time": marine["time"], "wave_height_max": marine["wave_height_max"]}
                ),
            )
        else:
            rec = EvidenceRecord.unknown(
                "Open-Meteo Marine API (ECMWF WAM)",
                why=wave_block.get("reason") or "marine data unavailable",
            )
        evidence.append(rec.to_dict())
        provenance["waves"] = rec.to_dict()

        # -- low-elevation screening --------------------------------------
        elev_block, elev_band = self._elevation_block(elevation)
        blocks["elevation_screening"] = elev_block
        if elev_block.get("status") == "ok":
            data_ok += 1
            rec = EvidenceRecord.open_data(
                elevation["source"],
                status=ClaimStatus.OBSERVED.value,
                temporal=TemporalClass.OBSERVED.value,
                location={"lat": lat, "lon": lon},
                method="Single-point DEM elevation; screening bands declared in the block.",
                limitations="Static DEM; screening only — not a flood or surge model.",
            )
            evidence.append(rec.to_dict())
            provenance["elevation_screening"] = rec.to_dict()

        # -- exposure -------------------------------------------------------
        blocks["exposure"] = self._exposure_block(osm_ctx)
        if blocks["exposure"].get("status") == "ok":
            data_ok += 1
            rec = EvidenceRecord.open_data(
                osm_ctx["source"],
                status=ClaimStatus.OBSERVED.value,
                temporal=TemporalClass.OBSERVED.value,
                location={"lat": lat, "lon": lon},
                method=f"Mapped OSM feature counts within {_OSM_RADIUS_M} m.",
                limitations=osm_ctx.get("note"),
            )
            evidence.append(rec.to_dict())
            provenance["exposure"] = rec.to_dict()

        # -- sea-level rise: PROJECTED/SCENARIO only, structurally separate --
        blocks["sea_level_rise"] = self._slr_block()
        slr_rec = EvidenceRecord.scientific(
            "IPCC AR6 WG1 Summary for Policymakers (2021), Table SPM.1",
            status=ClaimStatus.DOCUMENTED.value,
            temporal=TemporalClass.PROJECTED.value,
            link="https://www.ipcc.ch/report/ar6/wg1/",
            method=(
                "Approximate published 'likely' global-mean sea-level-rise ranges for "
                "2100 relative to 1995–2014, per emissions scenario (SSP1-2.6 / "
                "SSP2-4.5 / SSP5-8.5). Global means — local and regional values differ."
            ),
            confidence=Confidence.MEDIUM.value,
            limitations="Approximate ranges quoted from the AR6 SPM; not local projections.",
        )
        evidence.append(slr_rec.to_dict())
        provenance["sea_level_rise"] = slr_rec.to_dict()

        blocks["declared_limitations"] = (
            "Foundation analysis: NO storm-surge modelling, NO erosion rates, NO flood "
            "extents. Elevation screening is not a flood model. Sea-level-rise figures "
            "are scenario projections and never enter the observational blocks."
        )

        level = self._level(wave_pct, elev_band)
        if data_ok == 0:
            reasons = [
                b.get("reason") for b in (wave_block, elev_block, blocks["exposure"])
                if b.get("reason")
            ]
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Coastal analysis unavailable for this location.",
                level=None,
                blocks=blocks,
                evidence=evidence,
                provenance=provenance,
                unavailable_reason="; ".join(reasons) or "No coastal data obtained.",
            )

        status = "ok" if wave_block.get("status") == "ok" else "partial"
        return HazardAnalysis(
            hazard=self.id,
            location=location,
            status=status,
            summary=self._summary(wave_block, elev_block, level, status),
            level=level,
            blocks=blocks,
            evidence=evidence,
            provenance=provenance,
        )

    # -- blocks ----------------------------------------------------------

    @staticmethod
    def _waves_block(marine: Dict, today: date):
        """Wave block + percentile of the latest nowcast value (for the level)."""

        if "error" in marine:
            return {
                "status": "unavailable",
                "reason": marine["error"],
                "source": marine.get("source"),
            }, None

        times: List[str] = marine["time"]
        heights: List[Optional[float]] = marine["wave_height_max"]
        periods: List[Optional[float]] = marine["wave_period_max"]
        today_iso = today.isoformat()

        observed = [
            (t, h, p) for t, h, p in zip(times, heights, periods)
            if h is not None and t <= today_iso
        ]
        forecast = [
            {"date": t, "wave_height_max_m": h, "wave_period_max_s": p,
             "temporal": TemporalClass.FORECAST.value}
            for t, h, p in zip(times, heights, periods)
            if h is not None and t > today_iso
        ]
        if not observed:
            return {
                "status": "unavailable",
                "reason": "No observed/nowcast wave values in the marine series.",
            }, None

        last_date, last_h, last_p = observed[-1]
        pct = _series.percentile_rank([h for _t, h, _p in observed], last_h)
        rec_date, rec_h, _p = max(observed, key=lambda x: x[1])

        block = {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "latest": {
                "date": last_date,
                "wave_height_max_m": round(last_h, 2),
                "wave_period_max_s": round(last_p, 1) if last_p is not None else None,
                "temporal": TemporalClass.OBSERVED.value,
                "note": "Wave-model nowcast for the current window.",
            },
            "percentile_of_latest_vs_year": pct,
            "year_max": {"date": rec_date, "wave_height_max_m": round(rec_h, 2)},
            "forecast": forecast,
            "method": (
                "Percentile of the latest daily wave-height maximum within the past "
                "year of daily maxima at this point; forecast days are returned "
                "separately and labelled FORECAST."
            ),
            "source": marine["source"],
            "note": marine.get("note"),
        }
        return block, pct

    @staticmethod
    def _elevation_block(elevation: Dict):
        """Elevation screening block + the band label (for the level)."""

        if "error" in elevation:
            return {"status": "unavailable", "reason": elevation["error"]}, None
        elev = elevation.get("elevation_m")
        if elev is None:
            return {"status": "unavailable", "reason": "No elevation value returned."}, None
        band = "elevated"
        for limit, label in _ELEV_BANDS:
            if elev < limit:
                band = label
                break
        block = {
            "status": "ok",
            "claim_status": ClaimStatus.OBSERVED.value,
            "elevation_m": elev,
            "screening_band": band,
            "method": (
                "Low-elevation screening bands: <2 m very low-lying / <5 m low-lying / "
                "<10 m moderately low / else elevated. SCREENING ONLY — a static DEM "
                "value, not a flood, surge or inundation model."
            ),
            "source": elevation.get("source"),
        }
        return block, band

    @staticmethod
    def _exposure_block(osm_ctx: Dict) -> Dict[str, Any]:
        if "error" in osm_ctx:
            return {"status": "unavailable", "reason": osm_ctx["error"]}
        c = osm_ctx.get("counts") or {}
        return {
            "status": "ok",
            "claim_status": ClaimStatus.OBSERVED.value,
            "radius_m": osm_ctx.get("radius_m"),
            "buildings_mapped": c.get("buildings", 0),
            "roads_mapped": c.get("roads_all", 0),
            "critical_facilities_mapped": {
                "hospitals": c.get("hospitals", 0),
                "schools": c.get("schools", 0),
                "power_facilities": c.get("power_facilities", 0),
            },
            "note": (
                (osm_ctx.get("note") or "")
                + " Coastal infrastructure (ports, tourism, industry) is only visible "
                  "where mapped in OSM."
            ).strip(),
            "source": osm_ctx.get("source"),
        }

    @staticmethod
    def _slr_block() -> Dict[str, Any]:
        return {
            "status": "ok",
            "temporal": TemporalClass.PROJECTED.value,
            "claim_status": ClaimStatus.DOCUMENTED.value,
            "relative_to": "1995–2014",
            "scenarios": [
                {**s, "temporal": TemporalClass.SCENARIO.value} for s in _SLR_PROJECTIONS
            ],
            "source": _SLR_SOURCE,
            "note": (
                "Approximate published 'likely' global-mean sea-level-rise ranges by "
                "2100 (IPCC AR6 SPM Table SPM.1). Global means — local/regional values "
                "differ. Scenario projections: structurally separate from every "
                "observation in this analysis."
            ),
        }

    # -- level / summary ---------------------------------------------------

    @staticmethod
    def _level(wave_pct: Optional[float], elev_band: Optional[str]) -> Optional[HazardLevel]:
        if wave_pct is None and elev_band is None:
            return None
        low = elev_band in ("very low-lying", "low-lying")
        if wave_pct is not None:
            if (wave_pct >= 90 and low) or wave_pct >= 97:
                label = "High"
            elif wave_pct >= 90 or (wave_pct >= 75 and low):
                label = "Elevated"
            elif wave_pct >= 75:
                label = "Moderate"
            else:
                label = "Low"
            basis = (
                f"Screening indicator: latest wave-height percentile ({wave_pct}) vs the "
                f"past year, combined with the low-elevation screening band "
                f"({elev_band or 'unavailable'}). Declared rule: High = p≥90 on "
                f"low-lying land or p≥97 anywhere; Elevated = p≥90, or p≥75 on low-lying "
                f"land; Moderate = p≥75. NOT a validated predictor; no surge modelling."
            )
        else:
            label = "Elevated" if low else "Low"
            basis = (
                f"Screening indicator from elevation only ({elev_band}); wave data "
                f"unavailable. Low-elevation land near the coast is flagged Elevated. "
                f"NOT a validated predictor; no surge modelling."
            )
        return HazardLevel(label=label, basis=basis, validated=False)

    @staticmethod
    def _summary(wave_block, elev_block, level, status) -> str:
        parts: List[str] = []
        if wave_block.get("status") == "ok":
            latest = wave_block["latest"]
            parts.append(
                f"Wave height max {latest['wave_height_max_m']} m on {latest['date']} "
                f"(percentile {wave_block.get('percentile_of_latest_vs_year')} vs the past year)"
            )
        if elev_block.get("status") == "ok":
            parts.append(
                f"elevation {elev_block['elevation_m']} m ({elev_block['screening_band']})"
            )
        if level is not None:
            parts.append(f"screening level: {level.label}")
        if status == "partial":
            parts.append("partial: wave data unavailable")
        return "; ".join(parts) + "." if parts else "Coastal analysis complete."

    # -- map layers --------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        return [
            LayerSpec(
                layer_id="coastal.waves",
                label="Wave conditions (ECMWF WAM)",
                group="HAZARD",
                kind="points",
                endpoint="/api/v2/analyze?hazard=coastal&lat={lat}&lon={lon}",
                legend={"Low": "#22c55e", "Moderate": "#eab308", "Elevated": "#f97316", "High": "#ef4444"},
                source="Open-Meteo Marine API (ECMWF WAM)",
                url="https://open-meteo.com/en/docs/marine-weather-api",
                resolution="wave-model grid (tens of km)",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
                default_on=True,
            ).to_dict(),
            LayerSpec(
                layer_id="coastal.elevation",
                label="Low-elevation screening (DEM)",
                group="ENVIRONMENT",
                kind="points",
                endpoint="/api/v2/analyze?hazard=coastal&lat={lat}&lon={lon}",
                source="DEM (OpenTopoData EU-DEM 25 m / SRTM 90 m)",
                url="https://www.opentopodata.org/",
                resolution="25 m / 90 m",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
            ).to_dict(),
            LayerSpec(
                layer_id="coastal.slr",
                label="Sea-level rise scenarios (IPCC AR6)",
                group="PROJECTION",
                kind="raster",
                source="IPCC AR6 WG1 SPM Table SPM.1 (sourced constants)",
                url="https://www.ipcc.ch/report/ar6/wg1/",
                resolution="global mean",
                status="available",
                temporal=TemporalClass.PROJECTED.value,
                provenance={
                    "note": "Scenario projections (SSP1-2.6 / SSP2-4.5 / SSP5-8.5); "
                            "never mixed into observations."
                },
            ).to_dict(),
        ]
