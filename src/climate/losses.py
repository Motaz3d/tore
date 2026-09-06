"""
Loss Data Registry v2 (docs/ECONOMIC_INTELLIGENCE.md — no-fake-money rule).

Loads ``config/loss_registry.json`` — the registry of documented
disaster-loss data sources (EM-DAT, DesInventar, World Bank/GFDRR, NOAA,
Munich Re, Swiss Re) with their access conditions and integration status —
and formalises the loss summary with the platform's strict separation of
observed / estimated / modelled / projected figures.

Honesty contract (absolute):

- Every monetary figure carried by this module is documented by its source,
  tagged with claim_status, reference_period, geographic_scope and licence
  note. No figure is ever invented.
- Integrated free sources:
    * NOAA NCEI Billion-Dollar Weather and Climate Disasters (US only,
      1980-2021, public ArcGIS feature service) — aggregated to national
      totals from state-level costs/event counts.
- Staged-ingest sources (operator-provided export files; parsed when present,
  unavailable with reason when absent):
    * EM-DAT — drop ``data/emdat_export.csv`` downloaded from
      https://public.emdat.be after registration.
    * DesInventar — drop national export CSVs in ``data/desinventar_exports/``.
- Curated ``observed_events`` in the registry — a hand-maintained set of
  well-documented disaster events whose figures are published by
  official/primary sources. Each figure carries source, method, licence and
  limitations; events are matched to a queried location by country-scope
  bounding boxes (the figure stays a national/regional aggregate, never a
  point estimate).
- Commercial licences (Munich Re NatCatSERVICE, Swiss Re sigma) are planned
  after first platform revenue (operator decision 2026-09-02).
- Estimated, modelled and projected blocks are each ``not_available`` with
  an explicit statement — never merged, never invented.
- Registry sources are research candidates unless explicitly marked
  ``integrated`` or ``planned``; candidate records carry real official URLs
  and their access/licence conditions.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from ..dashboard.cache import cached
from .evidence import EvidenceRecord, utcnow_iso
from .ontology import ClaimStatus, Confidence

_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "loss_registry.json"
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_EMDAT_EXPORT_PATH = os.path.join(_DATA_DIR, "emdat_export.csv")
_DESINVENTAR_DIR = os.path.join(_DATA_DIR, "desinventar_exports")

_VALID_SOURCE_STATUS = ("candidate", "integrated", "planned", "unavailable")
_VALID_SOURCE_ACCESS = ("registration_required", "api", "download")

TTL_LOSSES = 24 * 3600.0  # 24 h — the NOAA state-level view changes slowly

OBSERVED_LOSSES_STATEMENT = "No documented loss figures in integrated sources."
ESTIMATED_LOSSES_STATEMENT = "No estimated loss figures exist in integrated sources."
MODELLED_LOSSES_STATEMENT = "No modelled loss figures exist in integrated sources."
PROJECTED_LOSSES_STATEMENT = "No projected loss figures exist in integrated sources."

_NOAA_COST_LAYER_URL = (
    "https://services3.arcgis.com/0Fs3HcaFfvzXvm7w/arcgis/rest/services/"
    "USA_Billion_Dollar_Disasters_view/FeatureServer/0/query"
)
_NOAA_FREQUENCY_LAYER_URL = (
    "https://services3.arcgis.com/0Fs3HcaFfvzXvm7w/arcgis/rest/services/"
    "USA_Billion_Dollar_Disasters_view/FeatureServer/1/query"
)

# Cost fields in layer 0 are in millions of USD (CPI-adjusted). Matching event-count fields.
_NOAA_COST_FIELDS = {
    "drought": ("Drought", "DroughtEvents"),
    "flooding": ("Flooding", "FloodingEvents"),
    "freeze": ("Freeze", "FreezeEvents"),
    "severe_storm": ("Severe Storm", "SevereStormEvents"),
    "tropical_cyclone": ("Tropical Cyclone", "TropicalCycloneEvents"),
    "wildfire": ("Wildfire", "WildfireEvents"),
    "winter_storm": ("Winter Storm", "WinterStormEvents"),
}

_USA_BBOX = {"min_lat": 18.0, "max_lat": 72.0, "min_lon": -180.0, "max_lon": -60.0}


def load_loss_registry(path: str | None = None) -> Dict[str, Any]:
    registry_path = path or os.environ.get("HYDRASHIELD_LOSS_REGISTRY") or _DEFAULT_REGISTRY
    with open(registry_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_loss_registry(registry: Dict[str, Any]) -> List[str]:
    """Structural validation; returns a list of problems (empty = valid)."""

    problems: List[str] = []
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        problems.append("no sources declared")
        sources = []
    seen: set = set()
    for i, src in enumerate(sources):
        sid = src.get("id")
        if not sid:
            problems.append(f"source {i}: missing id")
        elif sid in seen:
            problems.append(f"source {i}: duplicate id '{sid}'")
        seen.add(sid)
        for field in ("name", "provider", "url", "coverage", "status"):
            if not src.get(field):
                problems.append(f"source '{sid}': missing {field}")
        url = str(src.get("url") or "")
        if url and not url.startswith("https://"):
            problems.append(f"source '{sid}': url must be https")
        if src.get("access") not in _VALID_SOURCE_ACCESS:
            problems.append(
                f"source '{sid}': access must be one of {list(_VALID_SOURCE_ACCESS)}"
            )
        if src.get("status") not in _VALID_SOURCE_STATUS:
            problems.append(
                f"source '{sid}': status must be one of {list(_VALID_SOURCE_STATUS)}"
            )
    events = registry.get("observed_events")
    if not isinstance(events, list):
        problems.append("observed_events must be a list")
        events = []
    for i, ev in enumerate(events):
        evid = ev.get("id") or f"event {i}"
        for field in ("id", "name", "hazard", "reference_period"):
            if not ev.get(field):
                problems.append(f"observed_events '{evid}': missing {field}")
        areas = ev.get("affected_areas")
        if not isinstance(areas, list) or not areas:
            problems.append(
                f"observed_events '{evid}': affected_areas must be a non-empty list")
        else:
            for j, area in enumerate(areas):
                bbox = area.get("bbox")
                if not area.get("label") or not (
                        isinstance(bbox, list) and len(bbox) == 4
                        and all(isinstance(v, (int, float)) for v in bbox)):
                    problems.append(
                        f"observed_events '{evid}' area {j}: label and numeric bbox[4] required")
        figs = ev.get("figures")
        if not isinstance(figs, list) or not figs:
            problems.append(
                f"observed_events '{evid}': figures must be a non-empty list")
        else:
            for k, fig in enumerate(figs):
                for field in ("label", "value", "unit", "claim_status", "source",
                              "reference_period", "geographic_scope", "licence_note",
                              "provider_url", "method", "limitations"):
                    if fig.get(field) in (None, ""):
                        problems.append(
                            f"observed_events '{evid}' figure {k}: missing {field}")
                if fig.get("claim_status") != "DOCUMENTED":
                    problems.append(
                        f"observed_events '{evid}' figure {k}: claim_status must be "
                        "DOCUMENTED (curated observed events carry published figures only)")
                url = str(fig.get("provider_url") or "")
                if url and not url.startswith("https://"):
                    problems.append(
                        f"observed_events '{evid}' figure {k}: provider_url must be https")
    if not registry.get("separation_note"):
        problems.append("missing separation_note")
    return problems


def loss_sources() -> List[Dict[str, Any]]:
    """The registry's source records (integrated / planned / candidate)."""

    return list(load_loss_registry().get("sources") or [])


def load_observed_events() -> List[Dict[str, Any]]:
    """The registry's curated, documented observed-loss events."""

    return list(load_loss_registry().get("observed_events") or [])


def _point_in_bbox(lat: float, lon: float, bbox: List[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _event_matched_area(event: Dict[str, Any], lat: float, lon: float) -> Optional[str]:
    """The affected-area label whose country bbox contains the point, else None.

    Country bounding boxes overlap (a small country can sit inside a
    neighbour's bbox), so the SMALLEST containing bbox wins — the most
    specific country-scope match.
    """

    best: Optional[Tuple[float, str]] = None
    for area in event.get("affected_areas") or []:
        bbox = area.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if not _point_in_bbox(lat, lon, bbox):
            continue
        span = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if best is None or span < best[0]:
            best = (span, area.get("label") or "")
    return best[1] if best else None


def _curated_event_figures(
    for_lat: Optional[float] = None, for_lon: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Flatten curated registry events into documented loss figures.

    With a location, only events whose affected-area bounding boxes contain
    the point are included — a country-scope match; the figure stays a
    published national/regional aggregate, never a point estimate. Without a
    location, all curated events are included (registry-wide view).
    """
    events = load_observed_events()
    if not events:
        return [], "registry has no curated observed_events"

    figures: List[Dict[str, Any]] = []
    for event in events:
        area_label: Optional[str] = None
        if for_lat is not None and for_lon is not None:
            area_label = _event_matched_area(event, for_lat, for_lon)
            if area_label is None:
                continue
        for fig in event.get("figures") or []:
            entry = dict(fig)
            entry["event"] = event.get("name")
            entry["hazard"] = event.get("hazard")
            if area_label:
                entry["matched_area"] = area_label
            figures.append(entry)

    if not figures:
        return [], "no curated registry event covers this location"
    return figures, None


def _is_us_location(lat: float, lon: float) -> bool:
    """Coarse bounding-box test for US coverage (includes AK, HI, PR)."""
    return (
        _USA_BBOX["min_lat"] <= lat <= _USA_BBOX["max_lat"]
        and _USA_BBOX["min_lon"] <= lon <= _USA_BBOX["max_lon"]
    )


def _http_get_json(url: str, params: Dict[str, str], timeout: float = 30.0) -> Dict[str, Any]:
    """GET JSON from a URL with query parameters. Returns {"error": ...} on failure."""
    from urllib.parse import urlencode

    full = url + "?" + urlencode(params, doseq=True)
    req = urllib.request.Request(full, headers={"User-Agent": "Talaix-LossData/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code} from {url}"}
    except urllib.error.URLError as exc:
        return {"error": f"Network error: {exc.reason}"}
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON: {exc}"}
    except Exception as exc:
        return {"error": f"Fetch failed: {exc}"}
    return data


def _noaa_cost_out_fields() -> List[str]:
    """Layer 0 needs both cost fields and their matching event-count fields."""
    fields = ["STATE_NAME", "STATE_ABBR"]
    for cost_field, (_label, events_field) in _NOAA_COST_FIELDS.items():
        fields.extend([cost_field, events_field])
    return fields


@cached("losses_noaa_costs", TTL_LOSSES)
def _fetch_noaa_billions_state_costs() -> Dict[str, Any]:
    """Fetch NOAA state-level disaster costs (layer 0). Cached 24 h."""
    params = {
        "where": "1=1",
        "outFields": ",".join(_noaa_cost_out_fields()),
        "returnGeometry": "false",
        "resultRecordCount": "1000",
        "f": "json",
    }
    data = _http_get_json(_NOAA_COST_LAYER_URL, params)
    if "error" in data:
        return data
    features = data.get("features") or []
    if not features:
        return {"error": "NOAA feature service returned no cost features"}
    return {"features": features}


@cached("losses_noaa_frequency", TTL_LOSSES)
def _fetch_noaa_billions_state_frequency() -> Dict[str, Any]:
    """Fetch NOAA annual state-level disaster counts (layer 1). Cached 24 h."""
    params = {
        "where": "1=1",
        "outFields": ",".join(["STATE_NAME", "STATE_ABBR", "Year"] + list(_NOAA_COST_FIELDS.keys())),
        "returnGeometry": "false",
        "resultRecordCount": "1000",
        "f": "json",
    }
    data = _http_get_json(_NOAA_FREQUENCY_LAYER_URL, params)
    if "error" in data:
        return data
    features = data.get("features") or []
    if not features:
        return {"error": "NOAA feature service returned no frequency features"}
    return {"features": features}


def _parse_noaa_cost_features(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate state-level costs/event counts to US national totals."""
    totals: Dict[str, Dict[str, float]] = {"all": {"cost_millions": 0.0, "events": 0.0}}
    for hazard, (_label, events_field) in _NOAA_COST_FIELDS.items():
        totals[hazard] = {"cost_millions": 0.0, "events": 0.0}

    for feature in features:
        attr = feature.get("attributes") or {}
        for hazard, (_label, events_field) in _NOAA_COST_FIELDS.items():
            cost = attr.get(hazard)
            events = attr.get(events_field)
            if isinstance(cost, (int, float)):
                totals[hazard]["cost_millions"] += float(cost)
                totals["all"]["cost_millions"] += float(cost)
            if isinstance(events, (int, float)):
                totals[hazard]["events"] += float(events)
                totals["all"]["events"] += float(events)

    return totals


def _noaa_billions_figures() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return documented US national figures from NOAA, or ([], reason)."""
    cost_data = _fetch_noaa_billions_state_costs()
    if "error" in cost_data:
        return [], cost_data["error"]

    totals = _parse_noaa_cost_features(cost_data.get("features", []))
    all_cost = totals["all"]["cost_millions"]
    all_events = totals["all"]["events"]
    wildfire_cost = totals["wildfire"]["cost_millions"]
    wildfire_events = totals["wildfire"]["events"]

    if all_cost <= 0 and all_events <= 0:
        return [], "NOAA feature service returned zero loss totals"

    base = {
        "claim_status": ClaimStatus.DOCUMENTED.value,
        "source": "noaa_billions",
        "reference_period": "1980-2021",
        "geographic_scope": "United States",
        "licence_note": "US government public data; cite NOAA NCEI as source.",
        "provider_url": "https://www.ncei.noaa.gov/access/billions/",
        "method": "Aggregated from state-level costs/event counts in the public ArcGIS feature service USA_Billion_Dollar_Disasters_view; CPI-adjusted; state-level multi-state events may be summed across affected states.",
        "limitations": (
            "US-only. State-level source data: national totals are computed by Talaix "
            "as sums across states and may include double counting of events that "
            "affected multiple states. Reference period 1980-2021. No deaths data "
            "in this NOAA view."
        ),
    }

    figures: List[Dict[str, Any]] = [
        {
            "label": "Total US billion-dollar disaster costs",
            "value": round(all_cost / 1000.0, 2),
            "unit": "billion USD (CPI-adjusted)",
            **base,
        },
        {
            "label": "Total US billion-dollar disaster events",
            "value": int(all_events),
            "unit": "events",
            **base,
        },
        {
            "label": "US billion-dollar wildfire costs",
            "value": round(wildfire_cost / 1000.0, 2),
            "unit": "billion USD (CPI-adjusted)",
            **base,
        },
        {
            "label": "US billion-dollar wildfire events",
            "value": int(wildfire_events),
            "unit": "events",
            **base,
        },
    ]
    return figures, None


def _load_staged_emdat() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse an operator-provided EM-DAT CSV export if present."""
    if not os.path.exists(_EMDAT_EXPORT_PATH):
        return [], f"Staged EM-DAT export not found at {_EMDAT_EXPORT_PATH}"
    figures: List[Dict[str, Any]] = []
    try:
        with open(_EMDAT_EXPORT_PATH, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:
        return [], f"Failed to read EM-DAT export: {exc}"
    if not rows:
        return [], "EM-DAT export file is empty"

    # EM-DAT columns vary by export profile. We accept documented losses when present.
    deaths = 0
    for row in rows:
        try:
            deaths += int(row.get("Total Deaths") or row.get("Deaths") or 0)
        except (TypeError, ValueError):
            pass

    base = {
        "claim_status": ClaimStatus.DOCUMENTED.value,
        "source": "emdat",
        "reference_period": "operator-provided export",
        "geographic_scope": "Global (per EM-DAT export contents)",
        "licence_note": "Free for non-commercial research use after registration; redistribution restricted — check EM-DAT terms before integration.",
        "provider_url": "https://public.emdat.be/",
        "method": "Parsed from operator-downloaded EM-DAT CSV export in data/emdat_export.csv.",
        "limitations": "Figures depend on the operator's export selection and reference period; redistribution terms must be verified per EM-DAT licence.",
    }

    if deaths > 0:
        figures.append(
            {
                "label": "EM-DAT documented disaster deaths",
                "value": deaths,
                "unit": "deaths",
                **base,
            }
        )
    return figures, None


def _load_staged_desinventar() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse operator-provided DesInventar national CSV exports if present."""
    if not os.path.isdir(_DESINVENTAR_DIR):
        return [], f"Staged DesInventar exports directory not found at {_DESINVENTAR_DIR}"
    figures: List[Dict[str, Any]] = []
    files = sorted(f for f in os.listdir(_DESINVENTAR_DIR) if f.endswith(".csv"))
    if not files:
        return [], "No CSV files in data/desinventar_exports/"

    for fname in files:
        path = os.path.join(_DESINVENTAR_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception:
            continue
        if not rows:
            continue
        figures.append(
            {
                "label": f"DesInventar national events ({fname})",
                "value": len(rows),
                "unit": "records",
                "claim_status": ClaimStatus.DOCUMENTED.value,
                "source": "desinventar",
                "reference_period": "operator-provided export",
                "geographic_scope": f"National (from {fname})",
                "licence_note": "Open access downloads of national loss databases; per-country data maintained by national authorities — verify per-country terms.",
                "provider_url": "https://www.desinventar.org/",
                "method": "Parsed from operator-downloaded DesInventar CSV export.",
                "limitations": "Record counts only; detailed losses require per-export schema alignment.",
            }
        )
    if not figures:
        return [], "No parseable DesInventar CSV exports found"
    return figures, None


def documented_loss_figures(
    *, for_lat: Optional[float] = None, for_lon: Optional[float] = None
) -> Dict[str, Any]:
    """
    Return all documented loss figures available for the requested context.

    If ``for_lat/for_lon`` are supplied, only sources whose geographic scope
    covers that point are included (currently: NOAA only for US bounding box).
    Staged-ingest files are included when present regardless of location.
    """
    figures: List[Dict[str, Any]] = []
    reasons: List[str] = []

    # NOAA — live, US only.
    if for_lat is None or for_lon is None or _is_us_location(for_lat, for_lon):
        noaa_figs, noaa_reason = _noaa_billions_figures()
        figures.extend(noaa_figs)
        if noaa_reason:
            reasons.append(f"NOAA: {noaa_reason}")
    else:
        reasons.append("NOAA: queried location is outside United States coverage")

    # Curated registry events — documented, country-scope matched.
    curated_figs, curated_reason = _curated_event_figures(for_lat, for_lon)
    figures.extend(curated_figs)
    if curated_reason:
        reasons.append(f"Registry events: {curated_reason}")

    # EM-DAT — staged ingest.
    emdat_figs, emdat_reason = _load_staged_emdat()
    figures.extend(emdat_figs)
    if emdat_reason:
        reasons.append(f"EM-DAT: {emdat_reason}")

    # DesInventar — staged ingest.
    des_figs, des_reason = _load_staged_desinventar()
    figures.extend(des_figs)
    if des_reason:
        reasons.append(f"DesInventar: {des_reason}")

    if figures:
        return {
            "status": "ok",
            "figures": figures,
            "figure_count": len(figures),
            "sources": sorted({f["source"] for f in figures}),
            "generated_at": utcnow_iso(),
        }
    return {
        "status": "unavailable",
        "figures": [],
        "reason": "; ".join(reasons) if reasons else OBSERVED_LOSSES_STATEMENT,
        "generated_at": utcnow_iso(),
    }


def loss_summary_items() -> Dict[str, Any]:
    """
    Flat summary for ``GET /api/v2/losses/summary``.

    Contract:
        {"status":"ok","items":[{"label":str,"value":str,"unit":str,
          "source":str,"reference_period":str}...],"disclaimer":str}
    """
    figures_data = documented_loss_figures()
    if figures_data["status"] != "ok":
        return {
            "status": "unavailable",
            "items": [],
            "disclaimer": (
                f"No documented loss figures available. {figures_data.get('reason', '')} "
                "Talaix reports only observed/documented losses from integrated free sources; "
                "commercial loss databases are planned after first revenue."
            ).strip(),
        }

    items = []
    for fig in figures_data["figures"]:
        items.append(
            {
                "label": fig["label"],
                "value": str(fig["value"]),
                "unit": fig["unit"],
                "source": fig["source"],
                "reference_period": fig["reference_period"],
            }
        )
    return {
        "status": "ok",
        "items": items,
        "disclaimer": (
            "Figures are documented losses from integrated free sources and curated "
            "registry events. NOAA values are US-only national aggregates (1980-2021, "
            "CPI-adjusted) computed from state-level data and may include double counting "
            "of multi-state events. Curated registry events are published national or "
            "regional aggregates — not point-specific loss estimates. "
            "Commercial reinsurance loss databases are planned after first platform revenue."
        ),
    }


def loss_summary() -> Dict[str, Any]:
    """The loss summary with strict observed/estimated/modelled/projected
    separation. Observed losses now include real documented figures when
    available; other blocks remain not_available."""

    registry = load_loss_registry()
    sources = registry.get("sources") or []
    events = registry.get("observed_events") or []
    figures_data = documented_loss_figures()
    integrated = [s for s in sources if s.get("status") == "integrated"]
    planned = [s for s in sources if s.get("status") == "planned"]
    reviewed = [s.get("id") for s in sources if s.get("id")]

    if figures_data["status"] == "ok":
        observed_losses = {
            "status": "ok",
            "statement": f"{figures_data['figure_count']} documented loss figure(s) from "
                         "integrated free sources.",
            "figure_count": figures_data["figure_count"],
            "figures": figures_data["figures"],
            "sources_integrated": sorted(figures_data["sources"]),
            "sources_reviewed": reviewed,
            "confidence": Confidence.LOW.value,
        }
    else:
        observed_losses = {
            "status": "unavailable",
            "statement": OBSERVED_LOSSES_STATEMENT,
            "reason": figures_data.get("reason"),
            "sources_reviewed": reviewed,
            "confidence": Confidence.LOW.value,
        }

    by_status: Dict[str, int] = {}
    for s in sources:
        by_status[str(s.get("status"))] = by_status.get(str(s.get("status")), 0) + 1

    return {
        "status": "ok",
        "generated_at": utcnow_iso(),
        "observed_losses": observed_losses,
        "estimated_losses": {
            "status": "not_available",
            "statement": ESTIMATED_LOSSES_STATEMENT,
            "confidence": Confidence.LOW.value,
        },
        "modelled_losses": {
            "status": "not_available",
            "statement": MODELLED_LOSSES_STATEMENT,
            "confidence": Confidence.LOW.value,
        },
        "projected_losses": {
            "status": "not_available",
            "statement": PROJECTED_LOSSES_STATEMENT,
            "confidence": Confidence.LOW.value,
        },
        "separation_note": registry.get("separation_note"),
        "registry": {
            "registry_id": registry.get("registry_id"),
            "version": registry.get("version"),
            "config": "config/loss_registry.json",
            "source_count": len(sources),
            "sources_by_status": by_status,
            "observed_event_count": len(events),
            "integrated_sources": [s.get("id") for s in integrated],
            "planned_sources": [s.get("id") for s in planned],
        },
        "limitations": [
            "Observed losses are sourced from integrated free datasets and the "
            "curated registry of documented events; every figure carries source, "
            "reference period, geographic scope and licence note.",
            "NOAA values are US-only national aggregates (1980-2021) computed from "
            "state-level data; multi-state events may be summed across affected states.",
            "Curated registry events are published national or regional aggregates "
            "matched to a location by country scope — never point-specific estimates.",
            "Commercial loss datasets (Munich Re NatCatSERVICE, Swiss Re sigma) are "
            "planned after first platform revenue and are not used.",
        ],
    }


