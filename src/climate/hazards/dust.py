"""
Dust / sandstorm hazard plugin — registered with an honest unavailable
state until a real pipeline is wired in.

Candidate real sources (documented, live-checked 2026-08-18):

- **CAMS** (Copernicus Atmosphere Monitoring Service) — dust aerosol
  optical depth forecast at the analysis point. The fetch pipeline is
  WIRED (2026-09): ADS ``retrieve/v1`` job → NetCDF → nearest grid cell
  (0 h + 24 h lead), key-gated exactly like NASA FIRMS — it activates
  when ``CAMS_ADS_URL`` / ``CAMS_ADS_KEY`` are configured (see
  .env.example); without them the module answers honestly unavailable.
- **WMO SDS-WAS** (Sand and Dust Storm Warning Advisory and Assessment
  System, Barcelona Dust Centre) — regional dust model evaluation and
  guidance; reference source, not an integrated fetch path.

Wired source (2026-09, gradual engine wiring):

- **NASA EONET ``dustHaze``** — open dust & haze incidents worldwide (free,
  no key). Powers the ``events`` layer — current incident monitoring
  context only. Analysis activates with CAMS credentials.

Terminology discipline: "dust storm", "sandstorm", "Sirocco",
"Khamseen/Khamsin" and "desert dust transport" are regional/technical
terms for related but not identical phenomena — any future pipeline must
classify events by the terminology of the underlying source, not merge
them.

No medical claims are ever made; air-quality relevance is reported only
from authoritative measurements when a source is integrated.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from ..ontology import TemporalClass
from .base import HazardAnalysis, HazardModule, LayerSpec


class DustModule(HazardModule):
    id = "dust"
    name = "Dust / sandstorm"
    tagline = ("Desert dust transport, dust storms and sandstorms — "
               "transport, exposure and historical context. ")

    # -- availability (honest) ------------------------------------------

    def availability(self) -> Tuple[bool, Optional[str]]:
        from .. import cams
        if cams.credentials() is None:
            return False, ("Dust analysis requires CAMS (Copernicus Atmosphere "
                           "Monitoring Service) credentials (CAMS_ADS_URL / "
                           "CAMS_ADS_KEY) — not configured. The fetch pipeline "
                           "is wired (ADS retrieve/v1) and activates once the "
                           "key is present. No dust values are produced "
                           "without a real source.")
        return True, None

    def events_availability(self) -> Tuple[bool, Optional[str]]:
        return True, None

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "NASA EONET dust & haze incidents (live)": {
                "start": "current/ongoing incidents", "end": "present (live)"},
            "CAMS dust forecasts (candidate)": {
                "start": "per CAMS documentation", "end": "present"},
            "WMO SDS-WAS model guidance (candidate)": {
                "start": "per SDS-WAS documentation", "end": "present"},
        }

    # -- core operations --------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        available, reason = self.availability()
        if not available:
            return HazardAnalysis(
                hazard=self.id,
                location={"lat": lat, "lon": lon, "name": name},
                status="unavailable",
                summary="Dust analysis is not available yet.",
                unavailable_reason=reason,
            )

        from .. import cams
        from ..evidence import EvidenceRecord
        from ..ontology import ClaimStatus

        location = {"lat": lat, "lon": lon, "name": name}
        aod = cams.fetch_cams_dust_aod(lat, lon)
        if aod.get("key_required") or "error" in aod:
            rec = EvidenceRecord.unknown(
                "CAMS — Copernicus Atmosphere Monitoring Service",
                why=aod.get("error", "CAMS unavailable"))
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="CAMS dust data is unavailable right now.",
                blocks={"dust_aod": {"status": "unavailable",
                                     "reason": aod.get("error")}},
                evidence=[rec.to_dict()],
                provenance={"dust_aod": rec.to_dict()},
                unavailable_reason=aod.get("error"),
            )

        blocks = {
            "dust_aod": {
                "status": "ok",
                "claim_status": ClaimStatus.MODELLED.value,
                "dataset": aod["dataset"],
                "variable": aod["variable"],
                "date": aod["date"],
                "aod_analysis": aod["aod_analysis"],
                "band_analysis": aod["band_analysis"],
                "aod_lead24": aod["aod_lead24"],
                "band_lead24": aod["band_lead24"],
                "grid": aod["grid"],
                "note": aod["note"],
                "source": aod["source"],
            },
            "declared_limitations": (
                "CAMS modelled dust AOD at the nearest grid cell only: NO "
                "ground-level PM measurement, NO health or visibility "
                "assessment; screening labels are declared in "
                "src/climate/cams.py. Air-quality context belongs to measured "
                "sources (OpenAQ, national stations) once wired."
            ),
        }
        rec = EvidenceRecord.open_data(
            aod["source"],
            status=ClaimStatus.MODELLED.value,
            temporal="FORECAST",
            dataset=aod["dataset"],
            location=location,
            method=("ADS retrieve/v1 job → NetCDF → nearest grid cell per "
                    "lead time (0 h analysis + 24 h lead)."),
            limitations="Modelled aerosol optical depth, not a ground measurement.",
        )
        return HazardAnalysis(
            hazard=self.id,
            location=location,
            status="ok",
            summary=(f"CAMS dust AOD (550 nm) at this point: "
                     f"{aod['aod_analysis']} — {aod['band_analysis']} "
                     f"(24 h lead: {aod['aod_lead24']} — {aod['band_lead24']}). "
                     "Modelled, screening-level."),
            blocks=blocks,
            evidence=[rec.to_dict()],
            provenance={"dust_aod": rec.to_dict()},
        )

    # -- events (live: NASA EONET dustHaze incidents) ---------------------

    def events(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Current NASA EONET dust & haze incidents near a point.

        A ``year`` query asks for a historical dust-event archive — honestly
        unavailable (EONET covers open incidents; SDS-WAS archives are a
        candidate, not wired)."""
        if year is not None:
            return {
                "hazard": self.id,
                "status": "unavailable",
                "reason": ("A historical dust-event archive is not wired in — "
                           "EONET covers open incidents only and WMO SDS-WAS "
                           "records are a candidate source. No per-year history "
                           "is invented."),
                "events": [],
            }
        from ...dashboard import real_data as rd
        from ._gdacs import haversine_km

        feed = rd.fetch_eonet_dust_haze()
        if "error" in feed:
            return {"hazard": self.id, "status": "unavailable",
                    "reason": feed["error"], "events": []}
        radius = min(max(float(radius_km), 50.0), 3000.0)
        incidents = []
        for ev in feed["events"]:
            d = haversine_km(lat, lon, ev["lat"], ev["lon"])
            if d > radius:
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
            "hazard": self.id,
            "status": "ok",
            "radius_km": radius,
            "coverage": "NASA EONET open dust & haze incidents (worldwide, live)",
            "note": ("Incident-report monitoring context — not a dust-forecast "
                     "or air-quality measurement."),
            "source": feed["source"],
            "events": incidents,
        }

    # -- map layers -------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        _ok, reason = self.availability()
        return [
            LayerSpec(
                layer_id="dust.eonet",
                label="Dust & haze incidents (NASA EONET)",
                group="HAZARD",
                kind="points",
                endpoint="/api/v2/events?hazard=dust&lat={lat}&lon={lon}&radius_km=3000",
                source="NASA EONET — Earth Observatory Natural Event Tracker",
                url="https://eonet.gsfc.nasa.gov/",
                resolution="Latest reported position per incident",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
                provenance={"note": ("Open dust & haze incident monitoring — not a "
                                     "dust forecast or air-quality measurement.")},
            ).to_dict(),
            LayerSpec(
                layer_id="dust.forecast",
                label="Dust aerosol forecast (CAMS)",
                group="HAZARD",
                kind="raster",
                source="CAMS — Copernicus Atmosphere Monitoring Service",
                url="https://ads.atmosphere.copernicus.eu/",
                status="available" if _ok else "key_required",
                temporal=TemporalClass.FORECAST.value,
                provenance={"note": reason if not _ok else (
                    "ADS retrieve/v1 pipeline wired — dust AOD at the analysis "
                    "point (0 h + 24 h lead), screening labels declared.")},
            ).to_dict(),
            LayerSpec(
                layer_id="dust.sds_was",
                label="Dust model guidance (WMO SDS-WAS)",
                group="EVIDENCE",
                kind="raster",
                source="WMO SDS-WAS — Barcelona Dust Centre",
                url="https://dust.aemet.es/",
                status="unavailable",
                temporal=TemporalClass.FORECAST.value,
                provenance={"note": "Candidate source — not integrated."},
            ).to_dict(),
        ]
