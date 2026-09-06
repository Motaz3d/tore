"""
Extreme-heat hazard plugin (Stage 4 foundation).

Real-data analyses only (ERA5 daily Tmax via the Open-Meteo archive):

- **Climatological percentile** of recent values against the same grid
  point's day-of-year climatology (±7-day pool, declared baseline period
  1991–2020 — the WMO standard normal, fully inside the ERA5 archive).
- **Heatwave spells** — runs of ≥3 consecutive days above the location's
  own day-of-year 90th percentile (declared, WMO-style method).
- **Historical comparison** — hottest events of the 1991–2020 baseline,
  taken from the ERA5 series itself (nothing imported from elsewhere).
- **Exposure** — mapped OSM buildings/critical facilities and WorldCover
  built-up land cover as population/urban proxies (declared as proxies).

Reanalysis values are model-assimilation output: labelled MODELLED, with
the archive lag (~5 days) stated.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..ontology import ClaimStatus, EvidenceClass, TemporalClass
from ..evidence import EvidenceRecord, content_hash
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec
from . import _series

_BASELINE = ("1991-01-01", "2020-12-31")   # WMO standard normal period
_RECENT_DAYS = 92
_ARCHIVE_LAG_DAYS = 5
_HEATWAVE_Q = 90.0                          # day-of-year 90th percentile threshold
_HEATWAVE_MIN_DAYS = 3                      # >= 3 consecutive days = one heatwave spell
_DOY_WINDOW = 7                             # ±7-day climatology pool
_OSM_RADIUS_M = 2000


class HeatModule(HazardModule):
    id = "heat"
    name = "Extreme heat"
    tagline = "Tmax percentile vs day-of-year climatology, heatwave spells, historical extremes."

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "ERA5 daily Tmax (Open-Meteo archive)": {
                "start": "1940",
                "end": "~5 days ago (archive lag)",
            },
        }

    # -- analysis ------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard import real_data as rd
        from ...dashboard import exposure as osm
        from ...gis_mapping.landcover import fetch_landcover

        today = date.today()
        recent_start = (today - timedelta(days=_RECENT_DAYS)).isoformat()
        archive_end = (today - timedelta(days=_ARCHIVE_LAG_DAYS)).isoformat()
        location = {"lat": lat, "lon": lon}

        clim = rd.fetch_daily_climate(lat, lon, _BASELINE[0], _BASELINE[1], ["temperature_2m_max"])
        recent = rd.fetch_daily_climate(lat, lon, recent_start, archive_end, ["temperature_2m_max"])
        osm_ctx = osm.fetch_osm_context(lat, lon, _OSM_RADIUS_M)
        landcover = fetch_landcover(lat, lon)

        blocks: Dict[str, Any] = {}
        evidence: List[Dict[str, Any]] = []
        provenance: Dict[str, Any] = {}

        if "error" in clim:
            rec = EvidenceRecord.unknown("Open-Meteo archive (ERA5)", why=clim["error"])
            evidence.append(rec.to_dict())
            provenance["climatology"] = rec.to_dict()
            blocks["current_vs_climatology"] = {"status": "unavailable", "reason": clim["error"]}
            blocks["exposure"] = self._exposure_block(osm_ctx, landcover)
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Heat analysis unavailable for this location.",
                blocks=blocks,
                evidence=evidence,
                provenance=provenance,
                unavailable_reason=clim["error"],
            )

        clim_times: List[str] = clim["time"]
        clim_vals: List[Optional[float]] = clim["temperature_2m_max"]

        # -- historical extremes from the baseline series itself ---------
        blocks["historical_extremes"] = self._extremes_block(clim_times, clim_vals)

        recent_ok = "error" not in recent and any(
            v is not None for v in recent.get("temperature_2m_max") or []
        )
        pct: Optional[float] = None
        if recent_ok:
            blocks["current_vs_climatology"], pct = self._current_block(
                recent, clim_times, clim_vals
            )
            blocks["heatwave_spells"] = self._spells_block(recent, clim_times, clim_vals)
        else:
            reason = recent.get("error") or "Recent series empty."
            blocks["current_vs_climatology"] = {"status": "unavailable", "reason": reason}
            blocks["heatwave_spells"] = {"status": "unavailable", "reason": reason}

        blocks["exposure"] = self._exposure_block(osm_ctx, landcover)

        # -- evidence ------------------------------------------------------
        rec = EvidenceRecord(
            EvidenceClass.OPEN_DATA_OFFICIAL.value,
            ClaimStatus.MODELLED.value,
            TemporalClass.HISTORICAL.value,
            clim["source"],
            dataset="ERA5 daily maximum 2 m temperature",
            provider_url="https://open-meteo.com/en/docs/historical-weather-api",
            link=clim.get("request_url"),
            location={"lat": lat, "lon": lon},
            reference_period={"start": _BASELINE[0], "end": _BASELINE[1]},
            method=(
                f"Percentile of recent Tmax vs ±{_DOY_WINDOW}-day day-of-year "
                f"climatology pool, baseline {_BASELINE[0][:4]}–{_BASELINE[1][:4]}; "
                f"heatwave = ≥{_HEATWAVE_MIN_DAYS} consecutive days above the location's "
                f"own day-of-year {_HEATWAVE_Q:.0f}th percentile (WMO-style). "
                f"Archive lag ~{_ARCHIVE_LAG_DAYS} days."
            ),
            resolution="~25 km reanalysis grid",
            limitations="Reanalysis, not a station measurement; archive lag ~5 days.",
            content_hash=content_hash({"time": clim_times, "temperature_2m_max": clim_vals}),
        )
        evidence.append(rec.to_dict())
        provenance["climatology"] = rec.to_dict()
        if recent_ok:
            rec2 = EvidenceRecord(
                EvidenceClass.OPEN_DATA_OFFICIAL.value,
                ClaimStatus.MODELLED.value,
                TemporalClass.HISTORICAL.value,
                recent["source"],
                dataset="ERA5 daily maximum 2 m temperature",
                link=recent.get("request_url"),
                location={"lat": lat, "lon": lon},
                reference_period={"start": recent_start, "end": archive_end},
                method="Recent window analysed against the 1991–2020 day-of-year climatology.",
                resolution="~25 km reanalysis grid",
                limitations="Reanalysis; most recent days subject to archive lag.",
                content_hash=content_hash(
                    {"time": recent["time"], "temperature_2m_max": recent["temperature_2m_max"]}
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
                method="OSM feature counts + WorldCover built-up fraction (proxies).",
                limitations="OSM completeness varies; proxies, not population counts.",
            )
            evidence.append(rec3.to_dict())
            provenance["exposure"] = rec3.to_dict()

        level = self._level(pct, blocks.get("heatwave_spells") or {})
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
        vals: List[Optional[float]] = recent["temperature_2m_max"]
        valid = [(t, v) for t, v in zip(times, vals) if v is not None]
        if not valid:
            return {"status": "unavailable", "reason": "Recent Tmax series is empty."}, None
        last_date, last_val = valid[-1]
        pool = _series.doy_window_pool(clim_times, clim_vals, last_date, _DOY_WINDOW)
        pct = _series.percentile_rank(pool, last_val)
        block = {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "latest": {"date": last_date, "tmax_c": round(last_val, 1)},
            "percentile_vs_doy_climatology": pct,
            "climatology_pool_size": len(pool),
            "method": (
                f"Percentile of the latest daily Tmax within the ±{_DOY_WINDOW}-day "
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
        vals: List[Optional[float]] = recent["temperature_2m_max"]
        thresholds = _series.doy_thresholds(
            times, clim_times, clim_vals, q=_HEATWAVE_Q, window_days=_DOY_WINDOW
        )
        spells = _series.detect_spells(
            times, vals, thresholds, min_len=_HEATWAVE_MIN_DAYS, above=True
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
                f"Heatwave spell = ≥{_HEATWAVE_MIN_DAYS} consecutive days with daily "
                f"Tmax above the location's own day-of-year {_HEATWAVE_Q:.0f}th "
                f"percentile (±{_DOY_WINDOW}-day pool, baseline {_BASELINE[0][:4]}–"
                f"{_BASELINE[1][:4]}), WMO-style. Window: last {_RECENT_DAYS} days."
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
            "hottest_days": [
                {"date": t, "tmax_c": round(v, 1)} for t, v in top
            ],
            "method": (
                f"Top-5 hottest days of the baseline series {_BASELINE[0][:4]}–"
                f"{_BASELINE[1][:4]} at this grid point (ERA5 daily Tmax)."
            ),
        }

    @staticmethod
    def _exposure_block(osm_ctx: Dict, landcover: Dict) -> Dict[str, Any]:
        if "error" in osm_ctx and "error" in landcover:
            return {
                "status": "unavailable",
                "reason": f"{osm_ctx.get('error')} | {landcover.get('error')}",
            }
        out: Dict[str, Any] = {
            "status": "ok",
            "claim_status": ClaimStatus.OBSERVED.value,
            "note": "Population/urban proxies only — not population counts.",
        }
        if "error" not in osm_ctx:
            c = osm_ctx.get("counts") or {}
            out["buildings_mapped"] = c.get("buildings", 0)
            out["critical_facilities_mapped"] = {
                "hospitals": c.get("hospitals", 0),
                "schools": c.get("schools", 0),
            }
            out["radius_m"] = osm_ctx.get("radius_m")
            out["source"] = osm_ctx.get("source")
        else:
            out["osm_reason"] = osm_ctx["error"]
        if "error" not in landcover:
            hist = landcover.get("histogram") or {}
            built = hist.get(50) or hist.get("50") or {}
            out["built_up_fraction"] = built.get("fraction", 0.0)
            out["landcover_source"] = landcover.get("source")
        else:
            out["landcover_reason"] = landcover["error"]
        out.setdefault("source", landcover.get("source") or osm_ctx.get("source"))
        return out

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
                f"Screening indicator: latest Tmax percentile vs the location's own "
                f"1991–2020 day-of-year climatology, banded <75 Low / 75–90 Moderate / "
                f"90–97 High / ≥97 Very high"
                f"{'; a heatwave spell is currently ongoing' if ongoing else ''}. "
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
                f"Latest Tmax {cur['latest']['tmax_c']}°C on {cur['latest']['date']} "
                f"(percentile {cur.get('percentile_vs_doy_climatology')} vs 1991–2020)"
            )
        spells = blocks.get("heatwave_spells") or {}
        if spells.get("status") == "ok":
            parts.append(
                f"{spells['count']} heatwave spell(s) in the window"
                + (" — one ongoing" if spells.get("ongoing") else "")
            )
        if level is not None:
            parts.append(f"screening level: {level.label}")
        if status == "partial":
            parts.append("partial: recent series unavailable")
        return "; ".join(parts) + "." if parts else "Heat analysis complete."

    # -- map layers --------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        return [
            LayerSpec(
                layer_id="heat.tmax_percentile",
                label="Tmax percentile vs climatology (ERA5)",
                group="HAZARD",
                kind="grid",
                endpoint="/api/v2/analyze?hazard=heat&lat={lat}&lon={lon}",
                legend={"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Very high": "#ef4444"},
                source="ERA5 daily Tmax (Open-Meteo archive), baseline 1991–2020",
                url="https://open-meteo.com/en/docs/historical-weather-api",
                resolution="~25 km reanalysis grid",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
                default_on=True,
            ).to_dict(),
            LayerSpec(
                layer_id="heat.exposure",
                label="Heat exposure proxies (OSM + WorldCover)",
                group="EXPOSURE",
                kind="points",
                endpoint="/api/v2/analyze?hazard=heat&lat={lat}&lon={lon}",
                source="OpenStreetMap (ohsome / Overpass); ESA WorldCover built-up",
                url="https://www.openstreetmap.org/",
                resolution="feature-level / 10 m",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
            ).to_dict(),
        ]
