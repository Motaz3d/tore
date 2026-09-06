"""
Talaix Press evidence-pack engine.

Deterministic, template-only composition for journalist-facing climate
evidence packs. No generative prose: every number comes from the verification
engine, the ERA5 climate-series fetcher, real satellite observations, or the
press-watch registry.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from ..dashboard.real_data import fetch_climate_series, fetch_satellite_data, reverse_geocode
from ..dashboard.site_image import build_site_context_png, site_context_caption
from .evidence import utcnow_iso
from .tx_seal import issue_seal
from .verification import VERIFICATION_HAZARDS, verify_asset

ENGINE_VERSION = "1.0.0"

SUPPORTED_LANGUAGES = {"en", "fr", "de"}
DEFAULT_LANGUAGE = "en"

_PRESS_WATCH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "press_watch.json"
)

# Severity ranking for topic auto-detection. Higher index = stronger story.
# The comparison uses the first token of the label, so compound labels such as
# "Extreme — active storm nearby" still rank correctly.
_LABEL_RANK = {
    "extreme": 8,
    "severe": 7,
    "very": 6,
    "high": 5,
    "elevated": 4,
    "moderate": 3,
    "mild": 2,
    "low": 1,
    "near": 0,
}


def _rank_label(label: Optional[str]) -> int:
    if not label:
        return -1
    token = str(label).lower().split()[0]
    return _LABEL_RANK.get(token, -1)


def _detect_topic(verification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the highest-severity checked hazard as the press lead topic."""
    checks = verification.get("hazard_checks") or []
    checks_by_hazard = {c.get("hazard"): c for c in checks}

    best: Optional[Dict[str, Any]] = None
    best_rank = -1
    for hazard_id in VERIFICATION_HAZARDS:
        check = checks_by_hazard.get(hazard_id)
        if not check:
            continue
        if check.get("claim_status") == "UNKNOWN":
            continue
        level = check.get("level") or {}
        rank = _rank_label(level.get("label"))
        if rank > best_rank:
            best_rank = rank
            best = {
                "hazard": hazard_id,
                "taxonomy_label": check.get("taxonomy_label"),
                "level": level.get("label"),
                "claim_status": check.get("claim_status"),
                "confidence": check.get("confidence"),
                "summary": check.get("summary"),
                "basis": level.get("basis"),
            }
    return best


def _place_name(lat: float, lon: float, name: Optional[str]) -> str:
    if name:
        return name.strip()
    geocoded = reverse_geocode(lat, lon)
    if "error" not in geocoded:
        return geocoded.get("name") or f"{lat:.4f}, {lon:.4f}"
    return f"{lat:.4f}, {lon:.4f}"


def load_press_watch() -> List[Dict[str, Any]]:
    """Return the curated press-watch registry, or an empty list if missing."""
    try:
        with open(_PRESS_WATCH_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("sources", []) if isinstance(data, dict) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Deterministic templates per language
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "en": {
        "headline": "Climate evidence pack: {place}",
        "subhead": "Topic: {taxonomy_label} — {level}",
        "lead": (
            "Talaix screened {place} ({lat:.3f}°N, {lon:.3f}°E) against six "
            "physical climate hazards using real Earth observation and ERA5 "
            "reanalysis data. The strongest signal is {taxonomy_label}, assessed "
            "as {level} ({claim_status}, confidence: {confidence})."
        ),
        "fact_baseline": (
            "The 1991–2020 baseline mean annual maximum temperature at this grid "
            "point was {base_tmax} °C."
        ),
        "fact_temp": (
            "The most recent complete year, {year}, was {sign}{anomaly} °C compared "
            "with that baseline."
        ),
        "fact_precip": (
            "Total precipitation in {year} was {precip_pct}% of the baseline annual "
            "total ({base_precip} mm)."
        ),
        "fact_hazard": (
            "{taxonomy_label}: {level} level — {summary}"
        ),
        "satellite_available": (
            "Latest cloud-free optical satellite observation on {date} reported NDVI "
            "{ndvi} (source: {source})."
        ),
        "satellite_unavailable": (
            "No recent cloud-free Sentinel-2 or Landsat observation is available "
            "for this point."
        ),
        "site_context": (
            "Site context image: {caption}"
        ),
        "honesty": (
            "All figures are generated from cited sources. Unavailable data is "
            "declared, not filled in. Levels are screening indicators unless "
            "labelled validated."
        ),
        "methodology": (
            "Hazard levels come from the Talaix verification engine. Temperature "
            "and precipitation anomalies are derived from the Open-Meteo archive "
            "(ERA5 / ERA5-Land). Satellite data comes from the Copernicus "
            "Sentinel-2 programme via the public Earth Search STAC catalog."
        ),
        "figure_climate_alt": "Annual mean of daily maximum temperature and precipitation anomaly chart.",
        "figure_ndvi_alt": "Sentinel-2 NDVI grid for the location.",
        "figure_site_alt": "Land cover and tree-cover-loss context image for the location.",
    },
    "fr": {
        "headline": "Dossier d’évidence climatique : {place}",
        "subhead": "Sujet : {taxonomy_label} — {level}",
        "lead": (
            "Talaix a analysé {place} ({lat:.3f}°N, {lon:.3f}°E) sur six risques "
            "climatiques physiques à partir d’observations terrestres réelles et "
            "de réanalyses ERA5. Le signal le plus fort est {taxonomy_label}, "
            "évalué à {level} ({claim_status}, confiance : {confidence})."
        ),
        "fact_baseline": (
            "La température maximale annuelle moyenne de référence 1991–2020 à ce "
            "point de grille était de {base_tmax} °C."
        ),
        "fact_temp": (
            "La dernière année complète, {year}, affiche un écart de {sign}{anomaly} °C "
            "par rapport à cette référence."
        ),
        "fact_precip": (
            "Les précipitations totales en {year} représentaient {precip_pct}% du "
            "total annuel de référence ({base_precip} mm)."
        ),
        "fact_hazard": (
            "{taxonomy_label} : niveau {level} — {summary}"
        ),
        "satellite_available": (
            "La dernière observation Sentinel-2 dégagée du {date} indique un NDVI "
            "de {ndvi} (source : {source})."
        ),
        "satellite_unavailable": (
            "Aucune observation Sentinel-2 dégagée récente n’est disponible pour ce point."
        ),
        "site_context": (
            "Image de contexte local : {caption}"
        ),
        "honesty": (
            "Tous les chiffres proviennent de sources citées. Les données "
            "indisponibles sont déclarées, jamais inventées. Les niveaux sont des "
            "indicateurs de dépistage sauf mention contraire."
        ),
        "methodology": (
            "Les niveaux de risque proviennent du moteur de vérification Talaix. "
            "Les anomalies de température et de précipitation sont dérivées de "
            "l’archive Open-Meteo (ERA5 / ERA5-Land). Les données satellites "
            "proviennent du programme Copernicus Sentinel-2 via le catalogue STAC "
            "public Earth Search."
        ),
        "figure_climate_alt": "Graphique annuel de la température maximale moyenne et de l’anomalie pluviométrique.",
        "figure_ndvi_alt": "Grille NDVI Sentinel-2 pour l’emplacement.",
        "figure_site_alt": "Image de contexte : couverture terrestre et perte de couvert arboré.",
    },
    "de": {
        "headline": "Klimabeweis-Paket: {place}",
        "subhead": "Thema: {taxonomy_label} — {level}",
        "lead": (
            "Talaix hat {place} ({lat:.3f}°N, {lon:.3f}°E) auf sechs physische "
            "Klimarisiken anhand realer Erdbeobachtungsdaten und ERA5-Reanalysen "
            "geprüft. Das stärkste Signal ist {taxonomy_label}, bewertet als "
            "{level} ({claim_status}, Konfidenz: {confidence})."
        ),
        "fact_baseline": (
            "Der mittlere jährliche Temperaturhöchstwert der 1991–2020-Baseline "
            "an diesem Gitterpunkt betrug {base_tmax} °C."
        ),
        "fact_temp": (
            "Das jüngste vollständige Jahr, {year}, lag {sign}{anomaly} °C über "
            "bzw. unter dieser Baseline."
        ),
        "fact_precip": (
            "Die Gesamtniederschläge im Jahr {year} lagen bei {precip_pct}% des "
            "jährlichen Baselinewerts ({base_precip} mm)."
        ),
        "fact_hazard": (
            "{taxonomy_label}: Stufe {level} — {summary}"
        ),
        "satellite_available": (
            "Die jüngste wolkenfreie Sentinel-2-Aufnahme vom {date} zeigte einen "
            "NDVI von {ndvi} (Quelle: {source})."
        ),
        "satellite_unavailable": (
            "Für diesen Punkt liegt keine aktuelle wolkenfreie Sentinel-2-Aufnahme vor."
        ),
        "site_context": (
            "Kontextbild des Standorts: {caption}"
        ),
        "honesty": (
            "Alle Angaben stammen aus genannten Quellen. Nicht verfügbare Daten "
            "werden offen ausgewiesen, nicht ergänzt. Die Stufen sind "
            "Screening-Indikatoren, sofern nicht anders gekennzeichnet."
        ),
        "methodology": (
            "Risikostufen stammen aus der Talaix-Verifizierungsengine. "
            "Temperatur- und Niederschlagsanomalien leiten sich aus dem "
            "Open-Meteo-Archiv (ERA5 / ERA5-Land) ab. Satellitendaten stammen "
            "aus dem Copernicus-Sentinel-2-Programm über den öffentlichen Earth-"
            "Search-STAC-Katalog."
        ),
        "figure_climate_alt": "Jahresdiagramm des mittleren Tageshöchstwerts und der Niederschlagsanomalie.",
        "figure_ndvi_alt": "Sentinel-2-NDVI-Raster für den Standort.",
        "figure_site_alt": "Kontextbild: Bodenbedeckung und Waldverlust am Standort.",
    },
}


def _template(lang: str, key: str) -> str:
    return _TEMPLATES.get(lang, _TEMPLATES[DEFAULT_LANGUAGE]).get(
        key, _TEMPLATES[DEFAULT_LANGUAGE][key]
    )


def _format_temperature_anomaly(anomaly: Optional[float]) -> Dict[str, str]:
    if anomaly is None:
        return {"sign": "", "anomaly": "n/a"}
    if anomaly >= 0:
        return {"sign": "+", "anomaly": f"{anomaly:.2f}"}
    return {"sign": "", "anomaly": f"{anomaly:.2f}"}


def _pack_id(location: Dict[str, Any], verification: Dict[str, Any], climate: Dict[str, Any]) -> str:
    payload = {
        "lat": round(float(location["lat"]), 4),
        "lon": round(float(location["lon"]), 4),
        "verification_id": verification.get("verification_id"),
        "climate_year": (climate.get("current") or {}).get("year"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def build_press_pack(lat: float, lon: float, name: Optional[str] = None, lang: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """
    Build a deterministic press evidence pack for a point.

    Returns a dict with ``ok`` True on success; failures are returned with
    ``ok`` False and an ``error`` key.
    """
    if lang not in SUPPORTED_LANGUAGES:
        return {"ok": False, "error": f"Unsupported language '{lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"}

    place = _place_name(lat, lon, name)

    try:
        verification = verify_asset(lat, lon, name=place)
    except Exception as exc:
        return {"ok": False, "error": f"Verification engine failed: {exc}"}

    topic = _detect_topic(verification)

    try:
        climate = fetch_climate_series(lat, lon)
    except Exception as exc:
        climate = {"error": str(exc)}

    try:
        satellite = fetch_satellite_data(lat, lon)
    except Exception as exc:
        satellite = {"error": str(exc)}

    # Core metadata
    location = {"lat": lat, "lon": lon, "name": place}
    pack_id = _pack_id(location, verification, climate)
    pack: Dict[str, Any] = {
        "ok": True,
        "pack_id": pack_id,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "language": lang,
        "tier": "public" if lang == DEFAULT_LANGUAGE else "subscriber",
        "location": location,
        "verification_id": verification.get("verification_id"),
        "topic": topic,
        "authenticity": issue_seal(
            "press",
            pack_id,
            {
                "lat": round(float(location["lat"]), 4),
                "lon": round(float(location["lon"]), 4),
                "verification_id": verification.get("verification_id"),
                "climate_year": (climate.get("current") or {}).get("year"),
            },
        ),
    }

    tmpl = lambda key: _template(lang, key)

    # Headline + lead
    headline = tmpl("headline").format(place=place)
    subhead = ""
    lead = ""
    if topic:
        subhead = tmpl("subhead").format(
            taxonomy_label=topic["taxonomy_label"],
            level=topic["level"] or "—",
        )
        lead = tmpl("lead").format(
            place=place,
            lat=abs(lat),
            lon=abs(lon),
            taxonomy_label=topic["taxonomy_label"],
            level=topic["level"] or "—",
            claim_status=topic["claim_status"] or "—",
            confidence=topic["confidence"] or "—",
        )
    pack["headline"] = headline
    pack["subhead"] = subhead
    pack["lead"] = lead

    # Key facts
    key_facts: List[str] = []
    climate_current = climate.get("current") or {}
    baseline = climate.get("baseline") or {}
    if "error" not in climate and baseline.get("mean_tmax_c") is not None:
        key_facts.append(
            tmpl("fact_baseline").format(base_tmax=f"{baseline['mean_tmax_c']:.2f}")
        )
        anomaly = _format_temperature_anomaly(climate_current.get("mean_tmax_anomaly_c"))
        key_facts.append(
            tmpl("fact_temp").format(
                year=climate_current.get("year", "—"),
                sign=anomaly["sign"],
                anomaly=anomaly["anomaly"],
            )
        )
        if climate_current.get("precip_pct_of_baseline") is not None:
            key_facts.append(
                tmpl("fact_precip").format(
                    year=climate_current.get("year", "—"),
                    precip_pct=f"{climate_current['precip_pct_of_baseline']:.1f}",
                    base_precip=f"{baseline.get('precip_mm', 0):.1f}",
                )
            )
    else:
        key_facts.append("Climate series unavailable for this point.")

    # Topic hazard fact
    if topic:
        key_facts.append(
            tmpl("fact_hazard").format(
                taxonomy_label=topic["taxonomy_label"],
                level=topic["level"] or "—",
                summary=topic["summary"] or "",
            )
        )

    # Satellite fact
    if "error" not in satellite and satellite.get("ndvi") is not None:
        key_facts.append(
            tmpl("satellite_available").format(
                date=satellite.get("observation_date", "—"),
                ndvi=f"{float(satellite['ndvi']):.3f}",
                source=satellite.get("source", "Sentinel-2"),
            )
        )
    else:
        key_facts.append(tmpl("satellite_unavailable"))

    pack["key_facts"] = key_facts

    # Quotable sourced lines (ready to attribute)
    quotable: List[Dict[str, Any]] = []
    if "error" not in climate:
        current_year = climate_current.get("year")
        anomaly = climate_current.get("mean_tmax_anomaly_c")
        base_tmax = baseline.get("mean_tmax_c")
        precip_pct = climate_current.get("precip_pct_of_baseline")
        if base_tmax is not None and current_year is not None:
            sign = "+" if anomaly and anomaly >= 0 else ""
            quotable.append({
                "text": (
                    f"The 1991–2020 baseline mean annual maximum temperature at this "
                    f"grid point was {base_tmax:.2f} °C; the most recent complete year "
                    f"({current_year}) was {sign}{anomaly if anomaly is not None else 'n/a'} °C "
                    f"compared with that baseline."
                ),
                "source": "Reanalysis (ERA5 / ERA5-Land via Open-Meteo archive)",
                "status": "MODELLED",
            })
        if precip_pct is not None and current_year is not None:
            quotable.append({
                "text": (
                    f"Total precipitation in {current_year} was {precip_pct:.1f}% of the "
                    f"1991–2020 baseline annual total."
                ),
                "source": "Reanalysis (ERA5 / ERA5-Land via Open-Meteo archive)",
                "status": "MODELLED",
            })
    if topic and topic.get("basis"):
        quotable.append({
            "text": topic["basis"],
            "source": f"Talaix verification engine — {topic['taxonomy_label']}",
            "status": topic.get("claim_status", "MODELLED"),
        })
    if "error" not in satellite and satellite.get("ndvi") is not None:
        quotable.append({
            "text": (
                f"Latest cloud-free Sentinel-2 observation on "
                f"{satellite.get('observation_date', '—')} reported NDVI "
                f"{float(satellite['ndvi']):.3f}."
            ),
            "source": satellite.get("source", "Sentinel-2 L2A (Earth Search STAC)"),
            "status": "OBSERVED",
        })
    pack["quotable_lines"] = quotable

    # Structured climate block (the machine-readable form of the anomaly
    # numbers used in the quotables — the page mini-table and API consumers
    # read this, so it must be populated, not just the prose).
    if "error" not in climate:
        pack["climate_block"] = {
            "baseline": baseline,
            "current": climate_current,
            "annual": climate.get("annual") or [],
            "source": climate.get("source"),
            "coverage_note": climate.get("coverage_note") or climate.get("note"),
        }
    else:
        pack["climate_block"] = {
            "error": climate.get("error"),
            "source": climate.get("source"),
            "declared_gap": True,
        }

    # Figures
    query = f"lat={lat:.4f}&lon={lon:.4f}"
    if name:
        query += f"&name={name}"
    pack["figures"] = [
        {
            "kind": "climate",
            "endpoint": f"/api/v2/press/figure/climate?{query}",
            "available": "error" not in climate and bool(climate.get("annual")),
            "alt_text": tmpl("figure_climate_alt"),
            "caption": (
                f"Annual mean of daily maximum temperature and precipitation anomaly "
                f"vs the {baseline.get('period', '1991–2020')} baseline."
            ),
        },
        {
            "kind": "ndvi",
            "endpoint": f"/api/v2/press/figure/ndvi?{query}",
            "available": "error" not in satellite and satellite.get("ndvi_grid") is not None,
            "alt_text": tmpl("figure_ndvi_alt"),
            "caption": (
                f"Sentinel-2 NDVI grid "
                f"({satellite.get('observation_date', 'recent')})."
            ),
        },
        {
            "kind": "site",
            "endpoint": f"/api/v2/press/figure/site?{query}",
            "available": build_site_context_png(lat, lon) is not None,
            "alt_text": tmpl("figure_site_alt"),
            "caption": site_context_caption(),
        },
    ]

    # Sources
    sources: List[Dict[str, str]] = []
    sources.append({
        "name": "ERA5 / ERA5-Land reanalysis",
        "provider": "Copernicus Climate Change Service / ECMWF",
        "url": "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels",
    })
    sources.append({
        "name": "Open-Meteo archive API",
        "provider": "Open-Meteo",
        "url": "https://open-meteo.com/en/docs/climate-api",
    })
    sources.append({
        "name": "Sentinel-2 Level-2A",
        "provider": "Copernicus / ESA",
        "url": "https://sentinel.esa.int/web/sentinel/missions/sentinel-2",
    })
    sources.append({
        "name": "Earth Search STAC catalog",
        "provider": "Element84",
        "url": "https://element84.com/earth-search/",
    })
    sources.append({
        "name": "ESA WorldCover 10 m",
        "provider": "ESA",
        "url": "https://worldcover.esa.int/",
    })
    sources.append({
        "name": "Hansen/UMD Global Forest Change",
        "provider": "University of Maryland / Google",
        "url": "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/download.html",
    })
    pack["sources"] = sources

    # Press watch
    pack["press_watch"] = load_press_watch()

    pack["honesty_note"] = tmpl("honesty")
    pack["methodology_note"] = tmpl("methodology")

    return pack
