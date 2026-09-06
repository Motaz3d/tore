"""
Volcanic hazard plugin — registered with an honest state: current-activity
monitoring live, eruption analysis/history honestly unavailable.

Wired source (2026-09 gradual engine wiring):

- **GDACS ``VO`` event feed** (UN-OCHA / EU JRC) — current volcanic-activity
  alerts worldwide (alert level, affected countries, validity window,
  warning centre). Free, no key. Powers the ``events`` layer — monitoring
  context only.

Authoritative historical source (documented, live-checked 2026-08-18):

- **Smithsonian / USGS Global Volcanism Program (GVP)** — the canonical
  database of Holocene and Pleistocene volcanoes and their documented
  eruptive history, plus weekly activity reports. The GVP site is
  bot-protected for automated clients (HTTP 403 to non-browser agents at
  check time); a manual/exported dataset path is required before
  integration, so no live fetch is wired in.

Discipline: Talaix does not predict eruptions. Volcanic intelligence here is
current-activity monitoring context; eruption analysis and historical
eruption archives stay honestly unavailable without a real source.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..ontology import TemporalClass
from .base import HazardAnalysis, HazardModule, LayerSpec

_EVENTS_RADIUS_KM = 3000.0


class VolcanicModule(HazardModule):
    id = "volcanic"
    name = "Volcanic activity"
    tagline = ("Documented volcanic activity and exposure — historical "
               "evidence and monitoring context; never eruption prediction.")

    # -- availability (honest) ------------------------------------------

    def availability(self) -> Tuple[bool, Optional[str]]:
        return False, ("Volcanic analysis is not yet integrated: the "
                       "authoritative source (Smithsonian GVP) requires a "
                       "manual/exported dataset path — automated access is "
                       "bot-protected. Current-activity monitoring IS live "
                       "via the GDACS VO feed (events layer). No volcanic "
                       "analysis values are produced without a real source.")

    def events_availability(self) -> Tuple[bool, Optional[str]]:
        return True, None

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "GDACS current volcanic-activity alerts (UN-OCHA / EU JRC)": {
                "start": "current/ongoing events", "end": "present (live)"},
            "Smithsonian GVP — documented eruptions (candidate)": {
                "start": "Holocene record", "end": "present (weekly reports) — not wired"},
        }

    # -- core operations (honest unavailable) ----------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        _available, reason = self.availability()
        return HazardAnalysis(
            hazard=self.id,
            location={"lat": lat, "lon": lon, "name": name},
            status="unavailable",
            summary="Volcanic analysis is not available yet.",
            unavailable_reason=reason,
        )

    # -- events (live: GDACS VO monitoring context) ----------------------

    def events(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        year: Optional[int] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Current GDACS volcanic-activity alerts near a point.

        A ``year`` query asks for the historical eruption archive — honestly
        unavailable until the GVP export path exists."""
        if year is not None:
            return {
                "hazard": self.id,
                "status": "unavailable",
                "reason": ("Historical eruption events require the Smithsonian "
                           "GVP database export — not yet integrated. GDACS "
                           "covers current alerts only, so no per-year history "
                           "is invented."),
                "events": [],
            }
        from ...dashboard import real_data as rd
        from ._gdacs import flatten_gdacs_event

        feed = rd.fetch_gdacs_volcanoes()
        if "error" in feed:
            return {"hazard": self.id, "status": "unavailable",
                    "reason": feed["error"], "events": []}
        radius = min(max(float(radius_km), 50.0), _EVENTS_RADIUS_KM)
        events = [e for e in
                  (flatten_gdacs_event(f, lat, lon, "VO") for f in feed["features"])
                  if e is not None and e["distance_km"] <= radius]
        events.sort(key=lambda e: e["distance_km"])
        return {
            "hazard": self.id,
            "status": "ok",
            "radius_km": radius,
            "coverage": "GDACS current volcanic-activity alerts (worldwide, live)",
            "note": ("Official volcanic-activity alert monitoring context — "
                     "not an eruption forecast."),
            "source": feed["source"],
            "events": events,
        }

    # -- map layers -------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        _ok, reason = self.availability()
        return [
            LayerSpec(
                layer_id="volcanic.gdacs_active",
                label="Current volcanic-activity alerts (GDACS monitoring)",
                group="HAZARD",
                kind="points",
                endpoint="/api/v2/events?hazard=volcanic&lat={lat}&lon={lon}&radius_km=3000",
                legend={"Red alert": "#ef4444", "Orange alert": "#f97316",
                        "Green alert": "#22c55e"},
                source="GDACS — Global Disaster Alert and Coordination System (UN-OCHA / EU JRC)",
                url="https://www.gdacs.org/",
                resolution="Latest official alert positions (warning-centre issues)",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
                provenance={"note": ("Current volcanic-activity alert monitoring "
                                     "context from the official warning centres "
                                     "via GDACS — not an eruption forecast.")},
            ).to_dict(),
            LayerSpec(
                layer_id="volcanic.gvp_events",
                label="Documented eruptions (Smithsonian GVP)",
                group="EVIDENCE",
                kind="points",
                source="Smithsonian Institution / USGS — Global Volcanism Program",
                url="https://volcano.si.edu/",
                status="unavailable",
                temporal=TemporalClass.HISTORICAL.value,
                provenance={"note": reason},
            ).to_dict(),
        ]
