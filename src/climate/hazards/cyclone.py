"""
Tropical-cyclone hazard plugin (hurricanes / typhoons / cyclones).

Real wired source (live-checked 2026-08-22):

- **GDACS** — Global Disaster Alert and Coordination System (UN-OCHA /
  EU JRC). The event-list API exposes current tropical-cyclone (``TC``)
  events worldwide as GeoJSON: position, alert level (Green/Orange/Red),
  affected countries, validity window and the originating warning centre
  (JTWC, NHC, …). Free, no key. Powers ``analyze`` (monitoring context
  around a point) and the active-storms map layer.

Documented archive (wired 2026-09 — prepared dataset):

- **NOAA IBTrACS** (International Best Track Archive for Climate
  Stewardship) — the analysis-grade global archive of historical cyclone
  tracks. The platform keeps a local copy of the ``last3years`` CSV
  (~10 MB, refreshed monthly) and serves documented tracks near a point
  (``events?year=…`` and the ``historical_tracks`` analysis block).
  Deeper history (since1980 / full archive) is a declared follow-up.

Discipline: Talaix does not predict cyclone tracks or landfall.
This module reports official monitoring context (where active storms are,
how far, at which alert level) and documented historical tracks — never an
invented forecast.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..ontology import ClaimStatus, EvidenceClass, TemporalClass
from ..evidence import EvidenceRecord, content_hash
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec

_NEAR_KM = 300.0        # "near" threshold for the screening level
_WATCH_KM = 500.0
_REGION_KM = 1000.0     # storms within this distance are listed
_EVENTS_RADIUS_KM = 3000.0   # default map-layer query radius cap


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _storm_record(feature: Dict[str, Any], lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Flatten one GDACS TC feature into the platform's event shape."""
    props = feature.get("properties") or {}
    # The SEARCH feed can carry non-TC event types alongside cyclones —
    # admit only genuine tropical-cyclone entries.
    if props.get("eventtype") != "TC":
        return None
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if len(coords) < 2:
        return None
    slon, slat = coords[0], coords[1]
    urls = props.get("url") or {}
    return {
        "id": f"gdacs-tc-{props.get('eventid')}-{props.get('episodeid')}",
        "name": props.get("name") or props.get("eventname") or "Tropical cyclone",
        "lat": float(slat),
        "lon": float(slon),
        "alert_level": props.get("episodealertlevel") or props.get("alertlevel"),
        "alert_score": props.get("episodealertscore", props.get("alertscore")),
        "from_date": props.get("fromdate"),
        "to_date": props.get("todate"),
        "countries": props.get("country") or "",
        "warning_centre": props.get("source") or "GDACS",
        "report_url": urls.get("report"),
        "distance_km": round(_haversine_km(lat, lon, float(slat), float(slon)), 1),
    }


class CycloneModule(HazardModule):
    id = "cyclone"
    name = "Tropical cyclones"
    tagline = ("Active hurricanes / typhoons / cyclones — official monitoring "
               "context and exposure; never track or landfall prediction.")

    # -- coverage ---------------------------------------------------------

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "GDACS active-storm monitoring (UN-OCHA / EU JRC)": {
                "start": "current/ongoing events", "end": "present (live)"},
            "NOAA IBTrACS historical tracks (prepared local copy)": {
                "start": "last 3 seasons", "end": "present — refreshed monthly"},
        }

    # -- analysis ---------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard import real_data as rd

        location = {"lat": lat, "lon": lon}
        feed = rd.fetch_active_cyclones()
        if "error" in feed:
            rec = EvidenceRecord.unknown(
                "GDACS — Global Disaster Alert and Coordination System",
                why=feed["error"])
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Cyclone monitoring is unavailable right now.",
                blocks={"active_monitoring": {"status": "unavailable",
                                              "reason": feed["error"]}},
                evidence=[rec.to_dict()],
                provenance={"active_monitoring": rec.to_dict()},
                unavailable_reason=feed["error"],
            )

        storms = [s for s in
                  (_storm_record(f, lat, lon) for f in feed["features"])
                  if s is not None]
        storms.sort(key=lambda s: s["distance_km"])
        regional = [s for s in storms if s["distance_km"] <= _REGION_KM]
        nearest = storms[0] if storms else None

        level = self._level(nearest)
        summary = self._summary(storms, nearest)
        blocks: Dict[str, Any] = {
            "active_monitoring": {
                "status": "ok",
                "claim_status": ClaimStatus.REPORTED.value,
                "temporal": TemporalClass.OBSERVED.value,
                "storms_monitored_worldwide": len(storms),
                f"within_{int(_REGION_KM)}_km": regional,
                "nearest": nearest,
                "method": ("Great-circle distance from the analysis point to each "
                           "active storm's latest official position; alert levels as "
                           "issued by the warning centre via GDACS."),
                "note": ("Official monitoring context only — no track, landfall or "
                         "impact forecast is made."),
                "source": feed["source"],
            },
        }

        # -- historical tracks (IBTrACS prepared archive, wired 2026-09) ----
        from .. import ibtracs
        hist = ibtracs.storms_near(lat, lon, radius_km=_REGION_KM)
        if "error" in hist:
            blocks["historical_tracks"] = {
                "status": "unavailable",
                "reason": hist["error"],
            }
        else:
            blocks["historical_tracks"] = {
                "status": "ok",
                "claim_status": ClaimStatus.DOCUMENTED.value,
                "temporal": TemporalClass.HISTORICAL.value,
                "seasons_covered": hist["coverage"]["seasons"],
                "storms_within_region": hist["total_matching"],
                "storms": [
                    {k: s[k] for k in ("sid", "name", "season", "basin",
                                       "max_wind_kt", "min_pres_mb",
                                       "peak_sshs", "closest_approach_km")}
                    for s in hist["storms"][:10]
                ],
                "method": ("IBTrACS best-track archive (prepared local copy, "
                           "last 3 seasons): storms whose track passes within "
                           f"{_REGION_KM:.0f} km; closest approach is the "
                           "great-circle minimum over all track points."),
                "note": ("Documented historical tracks — frequency context "
                         "only, never a seasonal forecast."),
                "source": ibtracs.IBTRACS_SOURCE,
            }

        rec = EvidenceRecord.open_data(
            feed["source"],
            status=ClaimStatus.REPORTED.value,
            temporal=TemporalClass.OBSERVED.value,
            location=location,
            method=("Current GDACS tropical-cyclone event list; per-storm "
                    "distance from the analysis point (haversine)."),
            limitations=("Positions are the warning centres' latest fixes; alert "
                         "levels are GDACS impact-oriented levels. Monitoring, "
                         "not a forecast."),
            link=feed.get("request_url"),
        )
        # Evidence content hash pins the exact storm set analysed.
        rec.content_hash = content_hash(
            {"storms": [(s["id"], s["lat"], s["lon"]) for s in storms]})
        evidence = [rec.to_dict()]
        provenance = {"active_monitoring": rec.to_dict()}
        if blocks["historical_tracks"].get("status") == "ok":
            hist_rec = EvidenceRecord.open_data(
                ibtracs.IBTRACS_SOURCE,
                status=ClaimStatus.DOCUMENTED.value,
                temporal=TemporalClass.HISTORICAL.value,
                location=location,
                method="IBTrACS best-track proximity query (closest approach over track points).",
                limitations="Prepared file covers the last 3 seasons only.",
            )
            hist_rec.content_hash = content_hash(
                {"sids": [s["sid"] for s in hist["storms"]]})
            evidence.append(hist_rec.to_dict())
            provenance["historical_tracks"] = hist_rec.to_dict()
        return HazardAnalysis(
            hazard=self.id,
            location=location,
            status="ok",
            summary=summary,
            level=level,
            blocks=blocks,
            evidence=evidence,
            provenance=provenance,
        )

    def _level(self, nearest: Optional[Dict[str, Any]]) -> HazardLevel:
        """Categorical monitoring level from distance + official alert level."""
        basis = ("Distance from the point to the nearest active storm and the "
                 "official GDACS alert level. Screening monitoring indicator — "
                 "not a forecast, not a landfall probability.")
        if nearest is None:
            return HazardLevel("No active cyclone monitored worldwide",
                               basis=basis + " No active storm in the GDACS feed.")
        d = nearest["distance_km"]
        red = (nearest.get("alert_level") or "").lower() == "red"
        if d <= _NEAR_KM:
            label = "Extreme — active storm nearby" if red else "High — active storm nearby"
        elif d <= _WATCH_KM:
            label = "High — storm in the wider area" if red else "Moderate — storm in the wider area"
        elif d <= _REGION_KM:
            label = "Moderate — storm in the region" if red else "Low — storm in the region"
        else:
            return HazardLevel("No active cyclone within 1,000 km",
                               basis=basis + f" Nearest active storm is {d:.0f} km away.")
        return HazardLevel(
            label,
            basis=(basis + f" Nearest: {nearest['name']} at {d:.0f} km, "
                           f"alert {nearest.get('alert_level') or 'n/a'} "
                           f"({nearest.get('warning_centre')})."))

    @staticmethod
    def _summary(storms: List[Dict[str, Any]], nearest: Optional[Dict[str, Any]]) -> str:
        if not storms:
            return ("No active tropical cyclone is currently monitored worldwide "
                    "(GDACS). Historical exposure is not assessed — the track "
                    "archive is not wired in.")
        assert nearest is not None
        near = [s for s in storms if s["distance_km"] <= _REGION_KM]
        if not near:
            return (f"{len(storms)} active tropical cyclone(s) monitored worldwide; "
                    f"the nearest ({nearest['name']}) is {nearest['distance_km']:.0f} km "
                    f"away. Monitoring context only — no forecast.")
        names = ", ".join(s["name"] for s in near[:3])
        return (f"{len(near)} active tropical cyclone(s) within 1,000 km: {names}. "
                f"Nearest: {nearest['name']} at {nearest['distance_km']:.0f} km, "
                f"alert {nearest.get('alert_level') or 'n/a'}. Monitoring context "
                f"only — no forecast.")

    # -- events (map layer feed) ------------------------------------------

    def events(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Active/ongoing cyclone events near a point (GDACS monitoring);
        historical season tracks from the IBTrACS prepared archive.

        A ``year`` query returns that season's documented tracks near the
        point (IBTrACS, last 3 seasons in the prepared file)."""
        if year is not None:
            from .. import ibtracs
            hist = ibtracs.storms_near(lat, lon, year=year, radius_km=max(
                min(float(radius_km), _EVENTS_RADIUS_KM), 50.0))
            if "error" in hist:
                return {"hazard": self.id, "status": "unavailable",
                        "reason": hist["error"], "events": []}
            return {
                "hazard": self.id,
                "status": "ok",
                "year": year,
                "coverage": (f"IBTrACS documented best tracks, season {year} "
                             f"(prepared file seasons {hist['coverage']['seasons'][0]}"
                             f"–{hist['coverage']['seasons'][-1]})"),
                "note": ("Documented historical tracks — closest approach over "
                         "all track points; never a seasonal forecast."),
                "source": ibtracs.IBTRACS_SOURCE,
                "events": hist["storms"],
                "total_matching": hist["total_matching"],
            }
        from ...dashboard import real_data as rd

        feed = rd.fetch_active_cyclones()
        if "error" in feed:
            return {"hazard": self.id, "status": "unavailable",
                    "reason": feed["error"], "events": []}
        radius = min(max(float(radius_km), 50.0), _EVENTS_RADIUS_KM)
        storms = [s for s in
                  (_storm_record(f, lat, lon) for f in feed["features"])
                  if s is not None and s["distance_km"] <= radius]
        storms.sort(key=lambda s: s["distance_km"])
        return {
            "hazard": self.id,
            "status": "ok",
            "radius_km": radius,
            "coverage": ("GDACS active/ongoing tropical-cyclone monitoring "
                         "(current events worldwide)"),
            "note": ("Monitoring positions from the official warning centres — "
                     "not historical tracks, not a forecast."),
            "source": feed["source"],
            "events": storms,
        }

    # -- map layers ---------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        return [
            LayerSpec(
                layer_id="cyclone.active",
                label="Active tropical cyclones (GDACS monitoring)",
                group="HAZARD",
                kind="points",
                endpoint="/api/v2/events?hazard=cyclone&lat={lat}&lon={lon}&radius_km=3000",
                legend={"Red alert": "#ef4444", "Orange alert": "#f97316",
                        "Green alert": "#22c55e"},
                source="GDACS — Global Disaster Alert and Coordination System (UN-OCHA / EU JRC)",
                url="https://www.gdacs.org/",
                resolution="Latest official storm positions (warning-centre fixes)",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
                provenance={"note": ("Active-storm monitoring positions and alert "
                                     "levels as issued by the warning centres "
                                     "(JTWC, NHC, …) via GDACS. Not a forecast.")},
            ).to_dict(),
            LayerSpec(
                layer_id="cyclone.ibtracs_tracks",
                label="Historical cyclone tracks (NOAA IBTrACS)",
                group="EVIDENCE",
                kind="geojson",
                endpoint="/api/v2/events?hazard=cyclone&lat={lat}&lon={lon}&radius_km=1000&year={year}",
                source="NOAA NCEI — International Best Track Archive (IBTrACS)",
                url="https://www.ncei.noaa.gov/products/international-best-track-archive",
                resolution="3-hourly best-track positions; prepared file covers the last 3 seasons",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
                provenance={"note": ("Documented best-track archive — the platform "
                                     "serves seasons from its prepared local copy "
                                     "(last 3 years, refreshed monthly); deeper "
                                     "history is a declared follow-up.")},
            ).to_dict(),
        ]
