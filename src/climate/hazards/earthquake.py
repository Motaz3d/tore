"""
Earthquake hazard plugin (wired 2026-09 — gradual engine wiring, wave 2).

Real wired sources (both free, no key, live-checked 2026-09-03):

- **USGS ANSS ComCat** (FDSN event web service) — the authoritative global
  earthquake catalogue: magnitude, depth, time, significance. Powers the
  analysis (documented seismicity context around a point) and the events
  layer.
- **EMSC-CSEM** (SeismicPortal FDSN event service) — independent second
  seismic source, always reported separately, never merged.

Discipline: Talaix does not predict earthquakes. This module reports
documented seismicity (what has been recorded near the point, how strong,
how recently) as screening context — never a probability of a future
earthquake, never a hazard-map claim. Long-term probabilistic seismic
hazard (GEM/OpenQuake, national maps) remains a documented candidate.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from ..ontology import ClaimStatus, EvidenceClass, TemporalClass
from ..evidence import EvidenceRecord, content_hash
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec
from ._gdacs import haversine_km

_RADIUS_KM = 500.0            # documented seismicity context radius
_DEEP_MIN_MAG = 4.0           # deep-history window magnitude floor
_DEEP_START_YEAR = 1973       # modern instrumental-catalog era start
_RECENT_MIN_MAG = 2.5         # recent-activity window magnitude floor
_SIGNIFICANT_MAG = 4.5        # "nearest significant event" threshold


class EarthquakeModule(HazardModule):
    id = "earthquake"
    name = "Earthquakes"
    tagline = ("Documented seismicity around a point — USGS ComCat + EMSC, "
               "reported per source; never an earthquake forecast.")

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "USGS ANSS ComCat (global catalogue)": {
                "start": "instrumental-catalog era", "end": "present (live)"},
            "EMSC-CSEM event service (second source)": {
                "start": "real-time catalogue", "end": "present (live)"},
            "GEM/OpenQuake probabilistic hazard (candidate)": {
                "start": "model releases", "end": "periodic — not wired"},
        }

    # -- analysis ---------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard import real_data as rd

        location = {"lat": lat, "lon": lon, "name": name}
        today = date.today()
        deep = rd.fetch_usgs_earthquakes(
            lat, lon, radius_km=_RADIUS_KM, min_magnitude=_DEEP_MIN_MAG,
            limit=200, start=f"{_DEEP_START_YEAR}-01-01", end=today.isoformat())
        recent = rd.fetch_usgs_earthquakes(
            lat, lon, radius_km=_RADIUS_KM, min_magnitude=_RECENT_MIN_MAG,
            limit=200, start=f"{today.year - 1}-01-01", end=today.isoformat())
        emsc = rd.fetch_emsc_earthquakes(lat, lon)

        if "error" in deep and "error" in recent:
            rec = EvidenceRecord.unknown(
                "USGS Earthquake Hazards (ANSS ComCat)",
                why=deep.get("error") or recent.get("error"))
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Earthquake catalogue unavailable right now.",
                blocks={"seismicity": {"status": "unavailable",
                                       "reason": deep.get("error") or recent.get("error")}},
                evidence=[rec.to_dict()],
                provenance={"seismicity": rec.to_dict()},
                unavailable_reason=deep.get("error") or recent.get("error"),
            )

        blocks: Dict[str, Any] = {}
        evidence: List[Dict[str, Any]] = []
        provenance: Dict[str, Any] = {}

        # -- documented seismicity (USGS ComCat) ---------------------------
        deep_events = [] if "error" in deep else deep["events"]
        recent_events = [] if "error" in recent else recent["events"]
        for e in deep_events + recent_events:
            e["distance_km"] = round(haversine_km(lat, lon, e["lat"], e["lon"]), 1)

        strongest = max(deep_events, key=lambda e: e["mag"], default=None)
        significant = sorted(
            (e for e in deep_events if e["mag"] >= _SIGNIFICANT_MAG),
            key=lambda e: e["distance_km"])[:5]
        nearest_recent = sorted(recent_events, key=lambda e: e["distance_km"])[:5]
        bands = {
            "2.5–3.9": sum(1 for e in recent_events if e["mag"] < 4.0),
            "4.0–4.9": sum(1 for e in recent_events if 4.0 <= e["mag"] < 5.0),
            "5.0+": sum(1 for e in recent_events if e["mag"] >= 5.0),
        }

        blocks["seismicity"] = {
            "status": "ok",
            "claim_status": ClaimStatus.DOCUMENTED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "radius_km": _RADIUS_KM,
            "deep_window": {
                "start": f"{_DEEP_START_YEAR}-01-01",
                "magnitude_floor": _DEEP_MIN_MAG,
                "events_in_window": len(deep_events),
                "note": ("Catalog query (latest 200 matching events) — counts "
                         "describe the catalog window, not a complete rate."),
            },
            "strongest_documented": (
                {"mag": strongest["mag"], "place": strongest["place"],
                 "time": strongest["time"], "depth_km": strongest["depth_km"],
                 "distance_km": strongest["distance_km"], "url": strongest["url"]}
                if strongest else None),
            "significant_events": significant,
            "recent_year": {
                "since": f"{today.year - 1}-01-01",
                "magnitude_floor": _RECENT_MIN_MAG,
                "events": len(recent_events),
                "by_magnitude_band": bands,
                "nearest": nearest_recent,
            },
            "source": ("USGS ANSS ComCat" if "error" not in deep else
                       "USGS ANSS ComCat (deep window unavailable; recent only)"),
            "note": ("Documented seismicity context — what has been recorded "
                     "near the point. NOT an earthquake forecast and not a "
                     "probabilistic hazard assessment."),
        }
        if "error" not in deep:
            rec = EvidenceRecord(
                EvidenceClass.OPEN_DATA_OFFICIAL.value,
                ClaimStatus.DOCUMENTED.value,
                TemporalClass.HISTORICAL.value,
                deep["source"],
                dataset="USGS ANSS Comprehensive Catalog (FDSN event web service)",
                provider_url="https://earthquake.usgs.gov/fdsnws/event/1/",
                link=deep.get("request_url"),
                location=location,
                reference_period={"start": f"{_DEEP_START_YEAR}-01-01",
                                  "end": today.isoformat()},
                method=(f"Catalog query: magnitude ≥ {_DEEP_MIN_MAG} within "
                        f"{_RADIUS_KM:.0f} km (deep window since {_DEEP_START_YEAR}), "
                        f"plus magnitude ≥ {_RECENT_MIN_MAG} for the recent year; "
                        "distances are great-circle from the analysis point."),
                resolution="Event hypocentres (catalog completeness varies by region)",
                limitations=("Catalog query windows are capped (latest 200 events "
                             "per query) — counts are window counts, not complete "
                             "rates. Never an earthquake forecast."),
                content_hash=content_hash(
                    {"ids": [e["id"] for e in deep_events],
                     "mags": [e["mag"] for e in deep_events]}),
            )
            evidence.append(rec.to_dict())
            provenance["seismicity"] = rec.to_dict()

        # -- second source (EMSC — reported separately, never merged) ------
        blocks["emsc_second_source"] = self._emsc_block(emsc, lat, lon)
        if blocks["emsc_second_source"].get("status") == "ok":
            rec = EvidenceRecord(
                EvidenceClass.OPEN_DATA_OFFICIAL.value,
                ClaimStatus.DOCUMENTED.value,
                TemporalClass.HISTORICAL.value,
                emsc["source"],
                dataset="EMSC-CSEM FDSN event service",
                provider_url="https://www.seismicportal.eu/",
                link=emsc.get("request_url"),
                location=location,
                method=("Independent second seismic source near the point; "
                        "reported separately from USGS ComCat, never merged."),
                resolution="Event hypocentres",
                limitations="Euro-Med focus; independent of USGS — reported per source.",
                content_hash=content_hash({"ids": [e["id"] for e in emsc["events"]]}),
            )
            evidence.append(rec.to_dict())
            provenance["emsc_second_source"] = rec.to_dict()

        blocks["declared_limitations"] = (
            "Documented seismicity context only: NO earthquake prediction, NO "
            "probabilistic seismic-hazard claim (GEM/OpenQuake and national "
            "hazard maps are documented candidates), NO ground-shaking or "
            "damage modelling."
        )

        level = self._level(strongest, recent_events)
        summary = self._summary(strongest, recent_events, level)
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

    @staticmethod
    def _emsc_block(emsc: Dict, lat: float, lon: float) -> Dict[str, Any]:
        if "error" in emsc:
            return {"status": "unavailable", "reason": emsc["error"],
                    "source": emsc.get("source")}
        events = []
        for e in emsc["events"]:
            e = dict(e)
            e["distance_km"] = round(haversine_km(lat, lon, e["lat"], e["lon"]), 1)
            events.append(e)
        events.sort(key=lambda e: e["distance_km"])
        return {
            "status": "ok",
            "claim_status": ClaimStatus.DOCUMENTED.value,
            "role": ("Independent second seismic source — reported separately "
                     "from USGS ComCat, never merged."),
            "events": events[:10],
            "count": len(events),
            "source": emsc["source"],
            "note": emsc.get("note"),
        }

    @staticmethod
    def _level(strongest: Optional[Dict], recent_events: List[Dict]) -> HazardLevel:
        """Screening level from documented seismicity — never a prediction."""
        basis = ("Screening indicator from documented seismicity within "
                 f"{_RADIUS_KM:.0f} km: strongest recorded event since "
                 f"{_DEEP_START_YEAR} (magnitude ≥ {_DEEP_MIN_MAG} window) and "
                 "recent-year activity (magnitude ≥ 2.5). Bands are declared; "
                 "this is NOT a probabilistic hazard assessment and NOT an "
                 "earthquake forecast.")
        strongest_mag = strongest["mag"] if strongest else 0.0
        recent_strong = sum(1 for e in recent_events if e["mag"] >= 4.0)
        if strongest_mag >= 6.0 or recent_strong >= 3:
            label = "High documented seismicity"
        elif strongest_mag >= 5.0 or recent_strong >= 1:
            label = "Moderate documented seismicity"
        elif strongest_mag >= 4.0 or recent_events:
            label = "Low documented seismicity"
        else:
            label = "No significant documented seismicity in window"
        return HazardLevel(label=label, basis=basis, validated=False)

    @staticmethod
    def _summary(strongest: Optional[Dict], recent_events: List[Dict],
                 level: HazardLevel) -> str:
        parts: List[str] = []
        if strongest:
            parts.append(
                f"Strongest documented event within {_RADIUS_KM:.0f} km since "
                f"{_DEEP_START_YEAR}: M{strongest['mag']:.1f} ({strongest['place']}, "
                f"{strongest['distance_km']:.0f} km away)")
        else:
            parts.append(f"No M{_DEEP_MIN_MAG:.0f}+ event documented within "
                         f"{_RADIUS_KM:.0f} km since {_DEEP_START_YEAR}")
        parts.append(f"{len(recent_events)} recent event(s) (M≥2.5) in the last "
                     f"~year; screening level: {level.label}. Documented context — "
                     "never a forecast.")
        return " ".join(parts)

    # -- events (map layer feed) -------------------------------------------

    def events(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Documented earthquakes near a point (USGS ComCat).

        A ``year`` query returns that year's documented events (the catalog
        supports bounded windows — real data, no invention)."""
        from ...dashboard import real_data as rd

        radius = min(max(float(radius_km), 50.0), 2000.0)
        start = f"{year}-01-01" if year else None
        end = f"{year}-12-31" if year else None
        feed = rd.fetch_usgs_earthquakes(
            lat, lon, radius_km=radius, min_magnitude=_RECENT_MIN_MAG,
            limit=300, start=start, end=end)
        if "error" in feed:
            return {"hazard": self.id, "status": "unavailable",
                    "reason": feed["error"], "events": []}
        events = []
        for e in feed["events"]:
            e = dict(e)
            e["distance_km"] = round(haversine_km(lat, lon, e["lat"], e["lon"]), 1)
            events.append(e)
        events.sort(key=lambda e: e["distance_km"])
        return {
            "hazard": self.id,
            "status": "ok",
            "radius_km": radius,
            "year": year,
            "coverage": ("USGS ANSS ComCat documented earthquakes "
                         f"(magnitude ≥ {_RECENT_MIN_MAG}"
                         + (f", year {year}" if year else ", latest catalog events") + ")"),
            "note": ("Documented catalogue events — monitoring/historical "
                     "context, never an earthquake forecast."),
            "source": feed["source"],
            "events": events,
        }

    # -- map layers ---------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        return [
            LayerSpec(
                layer_id="earthquake.usgs_recent",
                label="Documented earthquakes (USGS ComCat)",
                group="HAZARD",
                kind="points",
                endpoint="/api/v2/events?hazard=earthquake&lat={lat}&lon={lon}&radius_km=500",
                legend={"M ≥ 5": "#ef4444", "M 4–4.9": "#f97316", "M 2.5–3.9": "#eab308"},
                source="USGS Earthquake Hazards (ANSS ComCat)",
                url="https://earthquake.usgs.gov/fdsnws/event/1/",
                resolution="Event hypocentres (catalog completeness varies by region)",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
                provenance={"note": ("Documented catalogue events — monitoring/"
                                     "historical context, never a forecast.")},
            ).to_dict(),
            LayerSpec(
                layer_id="earthquake.emsc",
                label="Earthquakes — second source (EMSC)",
                group="EVIDENCE",
                kind="points",
                endpoint="/api/v2/analyze?hazard=earthquake&lat={lat}&lon={lon}",
                source="EMSC-CSEM real-time earthquake services (FDSN)",
                url="https://www.seismicportal.eu/",
                resolution="Event hypocentres (Euro-Med focus)",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
                provenance={"note": ("Independent second seismic source — reported "
                                     "separately from USGS, never merged.")},
            ).to_dict(),
        ]
