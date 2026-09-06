"""
Extreme-wind hazard plugin (Stage 4 foundation).

Real-data analyses only (ERA5 daily wind-gust maxima via the Open-Meteo
archive) — the same declared pattern as the heat module:

- **Climatological percentile** of recent gust maxima against the same grid
  point's day-of-year climatology (±7-day pool, baseline 1991–2020).
- **Storm spells** — runs of ≥3 consecutive days above the location's own
  day-of-year 90th-percentile gust (declared method).
- **Historical comparison** — windiest days of the 1991–2020 baseline, from
  the ERA5 series itself.
- **Exposure** — mapped OSM infrastructure context (power facilities,
  buildings) within the analysis radius.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..ontology import ClaimStatus, EvidenceClass, TemporalClass
from ..evidence import EvidenceRecord, content_hash
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec
from . import _series

_BASELINE = ("1991-01-01", "2020-12-31")
_RECENT_DAYS = 92
_ARCHIVE_LAG_DAYS = 5
_STORM_Q = 90.0                  # day-of-year 90th percentile gust threshold
_STORM_MIN_DAYS = 3              # >= 3 consecutive days = one storm spell
_DOY_WINDOW = 7
_OSM_RADIUS_M = 2000


class WindModule(HazardModule):
    id = "wind"
    name = "Extreme wind"
    tagline = "Wind-gust percentile vs climatology, storm spells, historical extremes."

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "ERA5 daily wind gusts (Open-Meteo archive)": {
                "start": "1940",
                "end": "~5 days ago (archive lag)",
            },
        }

    # -- analysis ------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard import real_data as rd
        from ...dashboard import exposure as osm

        today = date.today()
        recent_start = (today - timedelta(days=_RECENT_DAYS)).isoformat()
        archive_end = (today - timedelta(days=_ARCHIVE_LAG_DAYS)).isoformat()
        location = {"lat": lat, "lon": lon}

        clim = rd.fetch_daily_climate(lat, lon, _BASELINE[0], _BASELINE[1], ["wind_gusts_10m_max"])
        recent = rd.fetch_daily_climate(lat, lon, recent_start, archive_end, ["wind_gusts_10m_max"])
        osm_ctx = osm.fetch_osm_context(lat, lon, _OSM_RADIUS_M)

        blocks: Dict[str, Any] = {}
        evidence: List[Dict[str, Any]] = []
        provenance: Dict[str, Any] = {}

        if "error" in clim:
            rec = EvidenceRecord.unknown("Open-Meteo archive (ERA5)", why=clim["error"])
            evidence.append(rec.to_dict())
            provenance["climatology"] = rec.to_dict()
            blocks["current_vs_climatology"] = {"status": "unavailable", "reason": clim["error"]}
            blocks["exposure"] = self._exposure_block(osm_ctx)
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Wind analysis unavailable for this location.",
                blocks=blocks,
                evidence=evidence,
                provenance=provenance,
                unavailable_reason=clim["error"],
            )

        clim_times: List[str] = clim["time"]
        clim_vals: List[Optional[float]] = clim["wind_gusts_10m_max"]

        blocks["historical_extremes"] = self._extremes_block(clim_times, clim_vals)

        recent_ok = "error" not in recent and any(
            v is not None for v in recent.get("wind_gusts_10m_max") or []
        )
        pct: Optional[float] = None
        if recent_ok:
            blocks["current_vs_climatology"], pct = self._current_block(
                recent, clim_times, clim_vals
            )
            blocks["storm_spells"] = self._spells_block(recent, clim_times, clim_vals)
        else:
            reason = recent.get("error") or "Recent series empty."
            blocks["current_vs_climatology"] = {"status": "unavailable", "reason": reason}
            blocks["storm_spells"] = {"status": "unavailable", "reason": reason}

        blocks["exposure"] = self._exposure_block(osm_ctx)

        # -- evidence ------------------------------------------------------
        rec = EvidenceRecord(
            EvidenceClass.OPEN_DATA_OFFICIAL.value,
            ClaimStatus.MODELLED.value,
            TemporalClass.HISTORICAL.value,
            clim["source"],
            dataset="ERA5 daily maximum 10 m wind gust",
            provider_url="https://open-meteo.com/en/docs/historical-weather-api",
            link=clim.get("request_url"),
            location={"lat": lat, "lon": lon},
            reference_period={"start": _BASELINE[0], "end": _BASELINE[1]},
            method=(
                f"Percentile of recent gust maxima vs ±{_DOY_WINDOW}-day day-of-year "
                f"climatology pool, baseline {_BASELINE[0][:4]}–{_BASELINE[1][:4]}; "
                f"storm spell = ≥{_STORM_MIN_DAYS} consecutive days above the location's "
                f"own day-of-year {_STORM_Q:.0f}th percentile. Archive lag "
                f"~{_ARCHIVE_LAG_DAYS} days."
            ),
            resolution="~25 km reanalysis grid",
            limitations="Reanalysis, not an anemometer measurement; archive lag ~5 days.",
            content_hash=content_hash({"time": clim_times, "wind_gusts_10m_max": clim_vals}),
        )
        evidence.append(rec.to_dict())
        provenance["climatology"] = rec.to_dict()
        if recent_ok:
            rec2 = EvidenceRecord(
                EvidenceClass.OPEN_DATA_OFFICIAL.value,
                ClaimStatus.MODELLED.value,
                TemporalClass.HISTORICAL.value,
                recent["source"],
                dataset="ERA5 daily maximum 10 m wind gust",
                link=recent.get("request_url"),
                location={"lat": lat, "lon": lon},
                reference_period={"start": recent_start, "end": archive_end},
                method="Recent window analysed against the 1991–2020 day-of-year climatology.",
                resolution="~25 km reanalysis grid",
                limitations="Reanalysis; most recent days subject to archive lag.",
                content_hash=content_hash(
                    {"time": recent["time"], "wind_gusts_10m_max": recent["wind_gusts_10m_max"]}
                ),
            )
            evidence.append(rec2.to_dict())
            provenance["recent"] = rec2.to_dict()
        if blocks["exposure"].get("status") == "ok":
            rec3 = EvidenceRecord.open_data(
                blocks["exposure"]["source"],
                status=ClaimStatus.OBSERVED.value,
                temporal=TemporalClass.OBSERVED.value,
                location={"lat": lat, "lon": lon},
                method=f"Mapped OSM infrastructure counts within {_OSM_RADIUS_M} m.",
                limitations=blocks["exposure"].get("note"),
            )
            evidence.append(rec3.to_dict())
            provenance["exposure"] = rec3.to_dict()

        level = self._level(pct, blocks.get("storm_spells") or {})
        status = "ok" if recent_ok else "partial"
        return HazardAnalysis(
            hazard=self.id,
            location=location,
            status=status,
            summary=self._summary(blocks, level, status),
            level=level,
            blocks=blocks,
            evidence=evidence,
            provenance=provenance,
        )

    # -- blocks ----------------------------------------------------------

    def _current_block(
        self, recent: Dict, clim_times: List[str], clim_vals: List[Optional[float]]
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        times: List[str] = recent["time"]
        vals: List[Optional[float]] = recent["wind_gusts_10m_max"]
        valid = [(t, v) for t, v in zip(times, vals) if v is not None]
        if not valid:
            return {"status": "unavailable", "reason": "Recent gust series is empty."}, None
        last_date, last_val = valid[-1]
        pool = _series.doy_window_pool(clim_times, clim_vals, last_date, _DOY_WINDOW)
        pct = _series.percentile_rank(pool, last_val)
        block = {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "latest": {"date": last_date, "gust_max_kmh": round(last_val, 1)},
            "percentile_vs_doy_climatology": pct,
            "climatology_pool_size": len(pool),
            "method": (
                f"Percentile of the latest daily gust maximum within the ±{_DOY_WINDOW}-day "
                f"day-of-year pool of the baseline period {_BASELINE[0][:4]}–"
                f"{_BASELINE[1][:4]} ({len(pool)} values, same grid point)."
            ),
            "source": recent["source"],
            "note": f"ERA5 archive lag: latest analysed date is ~{_ARCHIVE_LAG_DAYS} days behind real time.",
        }
        return block, pct

    def _spells_block(
        self, recent: Dict, clim_times: List[str], clim_vals: List[Optional[float]]
    ) -> Dict[str, Any]:
        times: List[str] = recent["time"]
        vals: List[Optional[float]] = recent["wind_gusts_10m_max"]
        thresholds = _series.doy_thresholds(
            times, clim_times, clim_vals, q=_STORM_Q, window_days=_DOY_WINDOW
        )
        spells = _series.detect_spells(
            times, vals, thresholds, min_len=_STORM_MIN_DAYS, above=True
        )
        ongoing = bool(spells and spells[-1]["end"] == times[-1])
        return {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "spells": spells,
            "count": len(spells),
            "ongoing": ongoing,
            "method": (
                f"Storm spell = ≥{_STORM_MIN_DAYS} consecutive days with daily gust "
                f"maximum above the location's own day-of-year {_STORM_Q:.0f}th "
                f"percentile (±{_DOY_WINDOW}-day pool, baseline {_BASELINE[0][:4]}–"
                f"{_BASELINE[1][:4]}). Window: last {_RECENT_DAYS} days."
            ),
        }

    @staticmethod
    def _extremes_block(clim_times: List[str], clim_vals: List[Optional[float]]) -> Dict[str, Any]:
        valid = [(t, v) for t, v in zip(clim_times, clim_vals) if v is not None]
        top = sorted(valid, key=lambda tv: -tv[1])[:5]
        return {
            "status": "ok" if valid else "unavailable",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "windiest_days": [
                {"date": t, "gust_max_kmh": round(v, 1)} for t, v in top
            ],
            "method": (
                f"Top-5 windiest days of the baseline series {_BASELINE[0][:4]}–"
                f"{_BASELINE[1][:4]} at this grid point (ERA5 daily gust maxima)."
            ),
        }

    @staticmethod
    def _exposure_block(osm_ctx: Dict) -> Dict[str, Any]:
        if "error" in osm_ctx:
            return {"status": "unavailable", "reason": osm_ctx["error"]}
        c = osm_ctx.get("counts") or {}
        return {
            "status": "ok",
            "claim_status": ClaimStatus.OBSERVED.value,
            "radius_m": osm_ctx.get("radius_m"),
            "power_facilities_mapped": c.get("power_facilities", 0),
            "buildings_mapped": c.get("buildings", 0),
            "note": osm_ctx.get("note"),
            "source": osm_ctx.get("source"),
        }

    # -- level / summary ---------------------------------------------------

    @staticmethod
    def _level(pct: Optional[float], spells_block: Dict[str, Any]) -> Optional[HazardLevel]:
        if pct is None:
            return None
        if pct >= 97:
            label = "Very high"
        elif pct >= 90:
            label = "High"
        elif pct >= 75:
            label = "Moderate"
        else:
            label = "Low"
        ongoing = spells_block.get("ongoing")
        return HazardLevel(
            label=label,
            score=round(pct, 1),
            score_max=100.0,
            basis=(
                f"Screening indicator: latest gust percentile vs the location's own "
                f"1991–2020 day-of-year climatology, banded <75 Low / 75–90 Moderate / "
                f"90–97 High / ≥97 Very high"
                f"{'; a storm spell is currently ongoing' if ongoing else ''}. "
                f"NOT a validated predictor."
            ),
            validated=False,
        )

    @staticmethod
    def _summary(blocks: Dict[str, Any], level: Optional[HazardLevel], status: str) -> str:
        parts: List[str] = []
        cur = blocks.get("current_vs_climatology") or {}
        if cur.get("status") == "ok":
            parts.append(
                f"Latest gust max {cur['latest']['gust_max_kmh']} km/h on "
                f"{cur['latest']['date']} (percentile "
                f"{cur.get('percentile_vs_doy_climatology')} vs 1991–2020)"
            )
        spells = blocks.get("storm_spells") or {}
        if spells.get("status") == "ok":
            parts.append(
                f"{spells['count']} storm spell(s) in the window"
                + (" — one ongoing" if spells.get("ongoing") else "")
            )
        if level is not None:
            parts.append(f"screening level: {level.label}")
        if status == "partial":
            parts.append("partial: recent series unavailable")
        return "; ".join(parts) + "." if parts else "Wind analysis complete."

    # -- map layers --------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        return [
            LayerSpec(
                layer_id="wind.gust_percentile",
                label="Wind-gust percentile vs climatology (ERA5)",
                group="HAZARD",
                kind="grid",
                endpoint="/api/v2/analyze?hazard=wind&lat={lat}&lon={lon}",
                legend={"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Very high": "#ef4444"},
                source="ERA5 daily wind-gust maxima (Open-Meteo archive), baseline 1991–2020",
                url="https://open-meteo.com/en/docs/historical-weather-api",
                resolution="~25 km reanalysis grid",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
                default_on=True,
            ).to_dict(),
            LayerSpec(
                layer_id="wind.exposure",
                label="Infrastructure exposure (OSM)",
                group="EXPOSURE",
                kind="points",
                endpoint="/api/v2/analyze?hazard=wind&lat={lat}&lon={lon}",
                source="OpenStreetMap (ohsome / Overpass)",
                url="https://www.openstreetmap.org/",
                resolution="feature-level",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
            ).to_dict(),
        ]
