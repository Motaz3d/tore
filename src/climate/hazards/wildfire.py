"""
Wildfire hazard plugin — adapts the proven Talaix wildfire engine
(`src/dashboard/real_analysis.py`) to the multi-hazard contract.

This is a *wrapper*, not a rewrite: the working pipeline keeps producing
its native payload (exposed as ``raw`` for full backward compatibility),
and the plugin projects it into :class:`HazardAnalysis`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from ..ontology import TemporalClass
from ..evidence import upgrade_provenance_block
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec


class WildfireModule(HazardModule):
    id = "wildfire"
    name = "Wildfire"
    tagline = "Fire danger, fuel moisture, spread, exposure and historical fire events."

    # -- availability --------------------------------------------------

    def availability(self) -> Tuple[bool, Optional[str]]:
        return True, None  # core pipeline works key-free (degraded layers degrade honestly)

    def events_availability(self) -> Tuple[bool, Optional[str]]:
        if not os.environ.get("FIRMS_MAP_KEY"):
            return False, ("NASA FIRMS key (FIRMS_MAP_KEY) not configured — "
                           "per-year observed fire history unavailable; current "
                           "open incidents ARE live via NASA EONET (no key).")
        return True, None

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        # Per dataset documentation; the UI builds year lists from this.
        return {
            "ERA5 fire weather (Open-Meteo archive)": {"start": "1940", "end": "near-present"},
            "NASA FIRMS VIIRS": {"start": "2012", "end": "present"},
            "NASA FIRMS MODIS": {"start": "2000", "end": "present"},
            "Sentinel-2 (EO evidence)": {"start": "2017", "end": "present"},
        }

    # -- analysis --------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard.real_analysis import TalaixRealAnalyser  # lazy: heavy engine

        raw = TalaixRealAnalyser().analyse_point(lat, lon, name=name)
        if raw.get("error"):
            return HazardAnalysis(
                hazard=self.id,
                location={"lat": lat, "lon": lon, "name": name},
                status="unavailable",
                summary="Wildfire analysis unavailable for this location.",
                unavailable_reason=str(raw.get("error")),
                raw=raw,
            )

        risk = (raw.get("analysis") or {}).get("risk") or {}
        score = risk.get("baseline")
        level = None
        if score is not None:
            level = HazardLevel(
                label=str(risk.get("class") or "Unknown"),
                score=score,
                score_max=100.0,
                basis=(
                    "Composite screening indicator anchored to the Canadian FWI computed "
                    "from real daily weather; NOT a validated predictor "
                    "(see docs/EVIDENCE_ARCHITECTURE.md §7)."
                ),
                validated=False,
            )
        fwi = (raw.get("fire_danger") or {}).get("fwi")
        summary_parts = []
        if score is not None:
            summary_parts.append(f"Wildfire risk {score:.0f}/100 ({risk.get('class')})")
        if fwi is not None:
            summary_parts.append(f"FWI {fwi:.1f}")
        summary = "; ".join(summary_parts) or "Wildfire analysis complete."

        return HazardAnalysis(
            hazard=self.id,
            location=raw.get("location") or {"lat": lat, "lon": lon, "name": name},
            status="ok",
            summary=summary,
            level=level,
            blocks={
                "fire_danger": raw.get("fire_danger"),
                "fire_danger_trend": raw.get("fire_danger_trend"),
                "weather": raw.get("weather"),
                "terrain": raw.get("terrain"),
                "satellite": raw.get("satellite"),
                "landcover": raw.get("landcover"),
                "active_fires": raw.get("active_fires"),
                "risk_explanation": raw.get("risk_explanation"),
                "exposure": raw.get("exposure"),
                "change": raw.get("change"),
            },
            provenance=upgrade_provenance_block(raw.get("provenance") or {}),
            raw=raw,
        )

    # -- events ------------------------------------------------------------

    _EONET_EVENTS_RADIUS_KM = 500.0

    def events(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        from ..fire_events import derive_fire_events  # lazy: network + FIRMS

        result = derive_fire_events(lat=lat, lon=lon, radius_km=radius_km, year=year, **kw)
        result["eonet"] = self._eonet_section(lat, lon, year)
        return result

    def _eonet_section(self, lat: float, lon: float, year: Optional[int]) -> Dict[str, Any]:
        """NASA EONET open wildfire incidents near the point — an
        independent second event source (incident reports, not FIRMS
        detections); always reported separately, never merged."""
        from ...dashboard import real_data as rd
        from ._gdacs import haversine_km

        feed = rd.fetch_eonet_wildfires()
        if "error" in feed:
            return {"status": "unavailable", "reason": feed["error"],
                    "source": None, "events": []}
        radius = self._EONET_EVENTS_RADIUS_KM
        incidents = []
        for ev in feed["events"]:
            d = haversine_km(lat, lon, ev["lat"], ev["lon"])
            if d > radius:
                continue
            if year is not None and (ev.get("date") or "")[:4] != str(year):
                continue
            incidents.append({
                "id": ev["id"],
                "name": ev["title"],
                "lat": ev["lat"],
                "lon": ev["lon"],
                "latest_report_date": ev.get("date"),
                "magnitude_value": ev.get("magnitude_value"),
                "magnitude_unit": ev.get("magnitude_unit"),
                "distance_km": round(d, 1),
                "link": ev.get("link"),
            })
        incidents.sort(key=lambda e: e["distance_km"])
        return {
            "status": "ok",
            "source": feed["source"],
            "coverage": f"Open wildfire incidents worldwide (last ~60 days), within {radius:.0f} km",
            "note": feed.get("note"),
            "events": incidents,
        }

    # -- map layers --------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        events_ok, events_reason = self.events_availability()
        return [
            LayerSpec(
                layer_id="wildfire.danger_grid",
                label="Fire danger grid (FWI-based)",
                group="HAZARD",
                kind="geojson",
                endpoint="/api/risk-grid?south={s}&west={w}&north={n}&east={e}&n=6",
                legend={"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Extreme": "#ef4444"},
                source="Open-Meteo daily fire weather + EU-DEM/SRTM terrain; FWI (Van Wagner 1987)",
                url="https://open-meteo.com/",
                resolution="~4–8 km cells (grid n=6)",
                temporal=TemporalClass.OBSERVED.value,
                # Opt-in: the coloured square cells no longer switch on
                # automatically (operator feedback: the grid squares were
                # visually noisy around a selected place).
                default_on=False,
            ).to_dict(),
            LayerSpec(
                layer_id="wildfire.events",
                label="Historical fire events (NASA FIRMS)",
                group="EVIDENCE",
                kind="points",
                endpoint="/api/v2/events?hazard=wildfire&lat={lat}&lon={lon}&radius_km={r}&year={year}",
                source="NASA FIRMS (VIIRS/MODIS), per-sensor, never merged",
                url="https://firms.modaps.eosdis.nasa.gov/",
                resolution="375 m (VIIRS) / 1 km (MODIS)",
                status="available" if events_ok else "key_required",
                provenance={"note": events_reason} if events_reason else None,
                temporal=TemporalClass.HISTORICAL.value,
            ).to_dict(),
            LayerSpec(
                layer_id="wildfire.eonet",
                label="Open wildfire incidents (NASA EONET)",
                group="EVIDENCE",
                kind="points",
                endpoint="/api/v2/events?hazard=wildfire&lat={lat}&lon={lon}&radius_km={r}",
                source="NASA EONET — incident reports, independent of FIRMS, never merged",
                url="https://eonet.gsfc.nasa.gov/",
                resolution="Latest reported position per incident",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
            ).to_dict(),
            LayerSpec(
                layer_id="wildfire.ndmi",
                label="Vegetation moisture (NDMI, Sentinel-2 / Landsat)",
                group="ENVIRONMENT",
                kind="raster",
                source="Copernicus Sentinel-2 L2A via Element84 STAC; USGS/NASA Landsat C2 L2 fallback via Planetary Computer STAC",
                url="https://sentiwiki.copernicus.eu/web/s2-mission",
                resolution="10 m (Sentinel-2) / 30 m (Landsat fallback)",
                temporal=TemporalClass.OBSERVED.value,
            ).to_dict(),
        ]
