"""
Compound Risk Engine v1 (docs/PRODUCT_VISION.md, IPCC AR6 WG2 risk framework).

Detects compound-event signals at a point from REAL current hazard signals,
following the four-class typology of Zscheischler et al. (2020) — cited in
``config/research_registry.json`` as ``zscheischler2020typology``. The paper
is a typology, NOT an operational algorithm: what lives here is a declared,
qualitative detector on top of it. The IPCC AR6 WG2 compound/cascading risk
framing is cited as ``ipccar6wg2``.

Signal sources (LIGHT only — the heavy wildfire engine is never called here):

- drought / heat / wind / flood via the registry's per-hazard analyzers
  (ERA5-based, cheap; flood tolerated honestly when no modelled river exists).
- fire danger: computed directly from the ERA5 daily archive fetcher
  (``real_data.fetch_weather_archive``) + the Canadian FWI System
  (``prediction.fwi.compute_fwi_series``, Van Wagner 1987; EFFIS classes).
  Declared as a fire-danger signal, NOT a wildfire risk analysis.

Honesty contract (absolute):

- NO numeric compound score. The output is a qualitative signal list only —
  no invented metric.
- Every signal carries the real values it rests on, evidence records and a
  claim-status label (MODELLED for detections on modelled reanalysis signals,
  INFERRED for preconditioning mechanisms).
- ``spatially_compounding`` is NOT computable at point scale and is returned
  explicitly as ``not_computable`` — never faked.
- A hazard that errors reduces the signal set (recorded in
  ``hazards_unavailable``); it never crashes the assessment.

Additive: :func:`dependence_analysis` computes EMPIRICAL co-occurrence
frequencies (and lift) of the declared elevated indicators over the
location's own trailing daily series — descriptive only, NOT a fitted
dependence model, no causal claim, no significance testing, and a declared
small-count guard (``insufficient_data`` below 10 event days, never a
number). It is wired into the assessment as the ``dependence`` block.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..dashboard.cache import cached
from .evidence import EvidenceRecord, utcnow_iso
from .hazards import _series
from .ontology import ClaimStatus, Confidence, EvidenceClass, TemporalClass

TTL_COMPOUND = 3600.0  # 1 h

#: Light per-hazard analyzers run via the registry (wildfire is NOT among
#: them — the full wildfire engine is heavy; see _fire_danger_signal).
LIGHT_HAZARDS = ("drought", "heat", "wind", "flood")

#: All hazard ids this engine can signal on. "wildfire" here means the cheap
#: FWI fire-danger signal, never the full wildfire engine.
KNOWN_HAZARDS = LIGHT_HAZARDS + ("wildfire",)

#: Declared elevated thresholds — the only gates the qualitative detector
#: uses. Each is stated again in every signal that rests on it.
ELEVATED_THRESHOLDS = {
    "drought": "elevated when the worst 30/90/180-day standardized precipitation anomaly (z) is <= -1.0",
    "heat": "elevated when the latest Tmax percentile vs the 1991-2020 day-of-year climatology is >= 90",
    "wind": "elevated when the latest gust percentile vs the 1991-2020 day-of-year climatology is >= 90",
    "flood": "elevated when the river-discharge or extreme-precipitation percentile vs the location's own record is >= 90",
    "wildfire": "fire-danger signal elevated when the latest Canadian FWI value is >= 30 (between the EFFIS Moderate and High classes; declared screening threshold)",
}

_FWI_ELEVATED = 30.0          # declared fire-danger threshold (see above)
_FWI_SPELL_MIN_DAYS = 3       # a high-FWI spell = >= 3 consecutive days >= threshold
_DROUGHT_Z_ELEVATED = -1.0
_PCT_ELEVATED = 90.0
_TEMPORAL_WINDOW_DAYS = 90    # trailing window for temporally compounding detection
_FIRE_HISTORY_DAYS = 120      # ERA5 window for the FWI fire-danger signal
_ARCHIVE_LAG_DAYS = 5         # Open-Meteo archive lag behind real time

#: Declared spell-follows-spell pairs for temporally compounding detection.
_TEMPORAL_PAIRS = (
    ("drought", "heat"),
    ("drought", "wildfire"),
    ("heat", "wildfire"),
)

_SPATIALLY_NOT_COMPUTABLE_REASON = (
    "Spatially compounding events (multiple locations connected by shared "
    "impacts or systems) require spatially resolved impact data across "
    "locations; a single-point analysis cannot detect them. Not estimated."
)

_NO_SIGNAL_STATEMENT = (
    "No compound signal detected from the analysed hazards at this location "
    "in the current window. This is a declared qualitative detector over "
    "screening-level signals — absence of a signal is not evidence of "
    "absence of risk."
)

_NO_NUMERIC_SCORE_NOTE = (
    "No numeric compound score is computed: the output is a qualitative "
    "signal list only (no invented metric)."
)

_TYPOLOGY_BASIS = (
    "Declared qualitative detector on top of the four-class compound-event "
    "typology of Zscheischler et al. (2020) — multivariate, temporally "
    "compounding, preconditioned, spatially compounding "
    "(research_registry: zscheischler2020typology). The typology is a "
    "classification framework, not an operational algorithm; thresholds and "
    "windows used here are declared Talaix screening choices, not "
    "parameters from the paper."
)


# ---------------------------------------------------------------------------
# Light per-hazard signal extraction (shared with cascading.py — do not
# duplicate this logic elsewhere)
# ---------------------------------------------------------------------------


def _signal_skeleton(hazard: str) -> Dict[str, Any]:
    return {
        "hazard": hazard,
        "status": "unavailable",
        "level": None,
        "elevated": False,
        "elevated_basis": None,
        "values": {},
        "spells": [],
        "spell_kind": None,
        "spell_status": None,
        "evidence": [],
        "source": None,
        "summary": "",
        "unavailable_reason": None,
    }


def _drought_signal(analysis) -> Dict[str, Any]:
    sig = _signal_skeleton("drought")
    sig["status"] = analysis.status
    sig["summary"] = analysis.summary
    sig["unavailable_reason"] = analysis.unavailable_reason
    sig["evidence"] = list(analysis.evidence or [])
    if analysis.level is not None:
        sig["level"] = analysis.level.to_dict()
    blocks = analysis.blocks or {}
    deficit = (blocks.get("precipitation_deficit") or {})
    windows = deficit.get("windows") or {}
    w90 = windows.get("90") or {}
    soil = blocks.get("soil_moisture") or {}
    dry = blocks.get("dry_spells") or {}

    min_z = analysis.level.score if analysis.level is not None else None
    sig["values"] = {
        "min_standardized_anomaly": min_z,
        "level_label": analysis.level.label if analysis.level else None,
        "precipitation_90d": {
            "current_sum_mm": w90.get("current_sum_mm"),
            "deficit_mm": w90.get("deficit_mm"),
            "standardized_anomaly": w90.get("standardized_anomaly"),
            "period": w90.get("current_period"),
        } if w90.get("status") == "ok" else None,
        "soil_moisture": {
            "anomaly_m3m3": soil.get("anomaly_m3m3"),
            "percentile_vs_climatology": soil.get("percentile_vs_climatology"),
            "as_of": soil.get("as_of"),
        } if soil.get("status") == "ok" else None,
        "dry_spell_ongoing": dry.get("ongoing"),
    }
    if min_z is not None and min_z <= _DROUGHT_Z_ELEVATED:
        sig["elevated"] = True
        sig["elevated_basis"] = (
            f"worst standardized precipitation anomaly z={min_z} <= "
            f"{_DROUGHT_Z_ELEVATED} (declared threshold)"
        )
    if dry.get("status") == "ok":
        sig["spells"] = list(dry.get("spells_last_year") or [])
        sig["spell_kind"] = "dry_spell"
        sig["spell_status"] = {"ongoing": bool(dry.get("ongoing")),
                               "count_last_year": dry.get("count_last_year")}
    sig["source"] = "ERA5 / ERA5-Land daily (Open-Meteo archive)"
    return sig


def _percentile_signal(analysis, hazard: str, spells_key: str,
                       spell_kind: str, latest_key: str) -> Dict[str, Any]:
    """Shared extraction for heat / wind (same block shape)."""

    sig = _signal_skeleton(hazard)
    sig["status"] = analysis.status
    sig["summary"] = analysis.summary
    sig["unavailable_reason"] = analysis.unavailable_reason
    sig["evidence"] = list(analysis.evidence or [])
    if analysis.level is not None:
        sig["level"] = analysis.level.to_dict()
    blocks = analysis.blocks or {}
    cur = blocks.get("current_vs_climatology") or {}
    spells_block = blocks.get(spells_key) or {}

    pct = cur.get("percentile_vs_doy_climatology") if cur.get("status") == "ok" else None
    sig["values"] = {
        "percentile_vs_doy_climatology": pct,
        "latest": cur.get("latest") if cur.get("status") == "ok" else None,
        "level_label": analysis.level.label if analysis.level else None,
        "spell_ongoing": spells_block.get("ongoing"),
    }
    if pct is not None and pct >= _PCT_ELEVATED:
        sig["elevated"] = True
        sig["elevated_basis"] = (
            f"latest {latest_key} percentile {pct} >= {_PCT_ELEVATED:.0f} "
            f"vs the 1991-2020 day-of-year climatology (declared threshold)"
        )
    if spells_block.get("status") == "ok":
        sig["spells"] = list(spells_block.get("spells") or [])
        sig["spell_kind"] = spell_kind
        sig["spell_status"] = {"ongoing": bool(spells_block.get("ongoing")),
                               "count": spells_block.get("count")}
    sig["source"] = "ERA5 daily (Open-Meteo archive)"
    return sig


def _flood_signal(analysis) -> Dict[str, Any]:
    sig = _signal_skeleton("flood")
    sig["status"] = analysis.status
    sig["summary"] = analysis.summary
    sig["unavailable_reason"] = analysis.unavailable_reason
    sig["evidence"] = list(analysis.evidence or [])
    if analysis.level is not None:
        sig["level"] = analysis.level.to_dict()
    blocks = analysis.blocks or {}
    dis = blocks.get("river_discharge") or {}
    pr = blocks.get("extreme_precipitation") or {}

    dis_pct = dis.get("percentile_of_latest") if dis.get("status") == "ok" else None
    pr_pct = (pr.get("percentile_of_max_daily_vs_record")
              if pr.get("status") == "ok" else None)
    sig["values"] = {
        "discharge_percentile": dis_pct,
        "discharge_latest": dis.get("latest") if dis.get("status") == "ok" else None,
        "precipitation_percentile": pr_pct,
        "max_daily_precip_mm": pr.get("max_daily_precip_mm")
        if pr.get("status") == "ok" else None,
        "level_label": analysis.level.label if analysis.level else None,
    }
    breached = []
    if dis_pct is not None and dis_pct >= _PCT_ELEVATED:
        breached.append(f"river-discharge percentile {dis_pct}")
    if pr_pct is not None and pr_pct >= _PCT_ELEVATED:
        breached.append(f"extreme-precipitation percentile {pr_pct}")
    if breached:
        sig["elevated"] = True
        sig["elevated_basis"] = (
            " and ".join(breached)
            + f" >= {_PCT_ELEVATED:.0f} vs the location's own record (declared threshold)"
        )
    if dis.get("status") == "ok":
        sig["spells"] = list(dis.get("high_discharge_spells_last_year") or [])
        sig["spell_kind"] = "high_discharge"
        sig["spell_status"] = {"count_last_year": len(sig["spells"])}
    sig["source"] = "GloFAS river discharge + ERA5 precipitation (Open-Meteo)"
    return sig


def _fire_danger_signal(lat: float, lon: float) -> Dict[str, Any]:
    """Cheap fire-danger signal: ERA5 daily archive -> Canadian FWI System.

    This is deliberately NOT the full wildfire engine. Declared screening
    approximation: daily T_max / RH_mean / max wind / precipitation sum in
    place of noon-standard FWI inputs (same convention as
    src/dashboard/history.py).
    """

    sig = _signal_skeleton("wildfire")
    sig["spell_kind"] = "high_fwi"
    try:
        from ..dashboard import real_data as rd
        from ..prediction.fwi import compute_fwi_series
    except ImportError as exc:  # pragma: no cover - defensive
        sig["unavailable_reason"] = f"FWI dependencies unavailable: {exc}"
        return sig

    today = date.today()
    start = (today - timedelta(days=_FIRE_HISTORY_DAYS)).isoformat()
    end = (today - timedelta(days=_ARCHIVE_LAG_DAYS)).isoformat()
    try:
        archive = rd.fetch_weather_archive(lat, lon, start, end)
    except Exception as exc:
        sig["unavailable_reason"] = f"ERA5 fire-weather fetch failed: {exc}"
        return sig
    if "error" in archive:
        sig["unavailable_reason"] = archive["error"]
        return sig

    times = archive.get("time") or []
    tmax = archive.get("temperature_2m_max") or []
    rh = archive.get("relative_humidity_2m_mean") or []
    wind = archive.get("wind_speed_10m_max") or []
    rain = archive.get("precipitation_sum") or []
    series_in: List[Dict[str, Any]] = []
    for i, t in enumerate(times):
        if i >= len(tmax) or i >= len(rh) or i >= len(wind):
            break
        if tmax[i] is None or rh[i] is None or wind[i] is None:
            continue
        series_in.append({
            "date": t,
            "temp_c": float(tmax[i]),
            "rh_pct": float(rh[i]),
            "wind_kmh": float(wind[i]),
            "rain_mm": float(rain[i]) if i < len(rain) and rain[i] is not None else 0.0,
        })
    if len(series_in) < 5:
        sig["unavailable_reason"] = (
            "Insufficient ERA5 fire-weather days for an honest FWI series.")
        return sig

    fwi_days = compute_fwi_series(series_in)
    dates = [d.date for d in fwi_days]
    fwi_vals = [round(d.fwi, 1) for d in fwi_days]
    latest = fwi_days[-1]
    spells = _series.detect_spells(
        dates, fwi_vals, _FWI_ELEVATED, min_len=_FWI_SPELL_MIN_DAYS, above=True)

    sig["status"] = "ok"
    sig["summary"] = (
        f"Fire danger (FWI) {round(latest.fwi, 1)} on {latest.date} "
        f"({latest.danger_class}); ERA5 archive lag ~{_ARCHIVE_LAG_DAYS} days.")
    sig["values"] = {
        "fwi_latest": round(latest.fwi, 1),
        "fwi_latest_date": latest.date,
        "fwi_danger_class": latest.danger_class,
        "fwi_max_window": max(fwi_vals),
        "window": {"start": dates[0], "end": dates[-1]},
    }
    sig["level"] = {
        "label": latest.danger_class,
        "score": round(latest.fwi, 1),
        "basis": "Canadian FWI value mapped to EFFIS-style danger classes; "
                 "screening approximation (daily aggregates, not noon-standard "
                 "inputs). NOT the full wildfire engine.",
        "validated": False,
    }
    if latest.fwi >= _FWI_ELEVATED:
        sig["elevated"] = True
        sig["elevated_basis"] = (
            f"latest FWI {round(latest.fwi, 1)} >= {_FWI_ELEVATED:.0f} "
            f"(declared fire-danger threshold)"
        )
    sig["spells"] = spells
    sig["spell_status"] = {"count_window": len(spells),
                           "ongoing": bool(spells and spells[-1]["end"] == dates[-1])}
    sig["source"] = archive.get("source", "Reanalysis (ERA5 via Open-Meteo archive)")
    sig["evidence"] = [EvidenceRecord.modelled(
        sig["source"],
        method=(
            "Canadian Fire Weather Index (Van Wagner 1987; EFFIS danger "
            "classes) over the ERA5 daily archive. Screening approximation: "
            "T_max / RH_mean / max wind / 24 h precipitation in place of "
            "noon-standard inputs; northern-hemisphere day-length tables. "
            f"Archive lag ~{_ARCHIVE_LAG_DAYS} days. Fire danger is not fire "
            "occurrence."
        ),
        temporal=TemporalClass.HISTORICAL.value,
        dataset="ERA5 daily Tmax / RH / wind / precipitation",
        provider_url="https://open-meteo.com/en/docs/historical-weather-api",
        location={"lat": lat, "lon": lon},
        reference_period={"start": dates[0], "end": dates[-1]},
        resolution="~25 km reanalysis grid",
        limitations="Reanalysis, not station measurements; FWI codes seeded "
                    "with conventional startup values at the window start.",
        confidence=Confidence.MEDIUM.value,
    ).to_dict()]
    return sig


_EXTRACTORS = {
    "drought": _drought_signal,
    "heat": lambda a: _percentile_signal(a, "heat", "heatwave_spells", "heatwave", "Tmax"),
    "wind": lambda a: _percentile_signal(a, "wind", "storm_spells", "storm", "gust"),
    "flood": _flood_signal,
}


@cached("compound_light_signals", TTL_COMPOUND)
def extract_light_signals(
    lat: float,
    lon: float,
    hazards: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the LIGHT per-hazard analyzers and extract real current signals.

    Shared by the compound engine and the cascading graph — import this, do
    not duplicate. Per-hazard failures are tolerated honestly: the hazard is
    recorded in ``hazards_unavailable`` and dropped from the signal set.
    Cached 1 h.
    """

    from . import registry

    lat, lon = float(lat), float(lon)
    selected = list(hazards) if hazards else list(KNOWN_HAZARDS)
    signals: Dict[str, Any] = {}
    unavailable: List[Dict[str, str]] = []

    for hid in selected:
        if hid not in KNOWN_HAZARDS:
            unavailable.append({"hazard": str(hid),
                                "reason": f"Unknown hazard '{hid}' for compound analysis."})
            continue
        if hid == "wildfire":
            sig = _fire_danger_signal(lat, lon)
        else:
            module = registry.get(hid)
            if module is None:
                unavailable.append({"hazard": hid,
                                    "reason": "Hazard module not registered in this checkout."})
                continue
            try:
                analysis = module.analyze(lat, lon)
            except Exception as exc:  # tolerate hazards that fail
                unavailable.append({"hazard": hid, "reason": str(exc)})
                continue
            try:
                sig = _EXTRACTORS[hid](analysis)
            except Exception as exc:  # pragma: no cover - defensive
                unavailable.append({"hazard": hid,
                                    "reason": f"signal extraction failed: {exc}"})
                continue
        if sig["status"] == "unavailable":
            unavailable.append({
                "hazard": hid,
                "reason": sig.get("unavailable_reason") or "analysis unavailable",
            })
        else:
            signals[hid] = sig

    latest_dates = []
    for sig in signals.values():
        v = sig.get("values") or {}
        cand = v.get("fwi_latest_date")
        if cand is None and isinstance(v.get("latest"), dict):
            cand = v["latest"].get("date")
        if cand is None and isinstance(v.get("discharge_latest"), dict):
            cand = v["discharge_latest"].get("date")
        if cand is None and isinstance(v.get("soil_moisture"), dict):
            cand = v["soil_moisture"].get("as_of")
        if cand:
            latest_dates.append(str(cand))
    as_of = max(latest_dates) if latest_dates else None

    return {
        "location": {"lat": lat, "lon": lon},
        "generated_at": utcnow_iso(),
        "as_of": as_of,
        "signals": signals,
        "hazards_unavailable": unavailable,
        "thresholds": dict(ELEVATED_THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# Typology detectors (declared, qualitative — zscheischler2020typology)
# ---------------------------------------------------------------------------


def _merged_evidence(signals: Dict[str, Any], hazard_ids: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for hid in hazard_ids:
        sig = signals.get(hid) or {}
        out.extend(sig.get("evidence") or [])
    return out


def _detect_multivariate(signals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """≥2 hazards simultaneously above their declared elevated thresholds."""

    elevated = {hid: s for hid, s in signals.items() if s.get("elevated")}
    if len(elevated) < 2:
        return []
    ids = sorted(elevated)
    return [{
        "type": "multivariate",
        "hazards": ids,
        "values": {
            hid: {
                "elevated_basis": elevated[hid]["elevated_basis"],
                "level_label": (elevated[hid].get("level") or {}).get("label"),
                "values": elevated[hid].get("values"),
            }
            for hid in ids
        },
        "claim_status": ClaimStatus.MODELLED.value,
        "temporal": TemporalClass.OBSERVED.value,
        "basis": (
            "Multivariate compound signal (zscheischler2020typology): "
            f"{len(ids)} hazards simultaneously above their declared elevated "
            "thresholds at this location in the current analysis window."
        ),
        "method": "Co-occurrence of elevated per-hazard screening signals; "
                  "thresholds declared in the payload.",
        "evidence": _merged_evidence(signals, ids),
        "research": ["zscheischler2020typology", "ipccar6wg2"],
    }]


def _detect_temporal(
    signals: Dict[str, Any],
    as_of: Optional[str],
    window_days: int = _TEMPORAL_WINDOW_DAYS,
) -> List[Dict[str, Any]]:
    """A hazard spell following another within the trailing window."""

    end = _series.parse_date(as_of) if as_of else None
    if end is None:
        return []
    start = end - timedelta(days=window_days)
    out: List[Dict[str, Any]] = []

    for first, second in _TEMPORAL_PAIRS:
        sig_a = signals.get(first) or {}
        sig_b = signals.get(second) or {}
        spells_a = sig_a.get("spells") or []
        spells_b = sig_b.get("spells") or []
        if not spells_a or not spells_b:
            continue
        sequences: List[Dict[str, Any]] = []
        for sa in spells_a:
            end_a = _series.parse_date(sa.get("end"))
            if end_a is None or end_a < start:
                continue
            for sb in spells_b:
                start_b = _series.parse_date(sb.get("start"))
                end_b = _series.parse_date(sb.get("end"))
                if start_b is None or end_b is None:
                    continue
                if start_b <= end_a or end_b > end:
                    continue
                sequences.append({
                    "preceding": {"hazard": first,
                                  "spell_kind": sig_a.get("spell_kind"),
                                  "spell": sa},
                    "following": {"hazard": second,
                                  "spell_kind": sig_b.get("spell_kind"),
                                  "spell": sb},
                    "gap_days": (start_b - end_a).days,
                })
        if not sequences:
            continue
        sequences.sort(key=lambda s: (s["following"]["spell"].get("start"), s["gap_days"]))
        out.append({
            "type": "temporally_compounding",
            "hazards": [first, second],
            "sequences": sequences[:5],
            "sequence_count": len(sequences),
            "window": {"start": start.isoformat(), "end": end.isoformat(),
                       "days": window_days},
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "basis": (
                "Temporally compounding signal (zscheischler2020typology): a "
                f"{sig_a.get('spell_kind')} spell was followed by a "
                f"{sig_b.get('spell_kind')} spell within the trailing "
                f"{window_days}-day window (real dates reported)."
            ),
            "method": "Spell pairs from the daily series where the following "
                      "spell starts after the preceding spell ends; both "
                      "inside the trailing window.",
            "evidence": _merged_evidence(signals, [first, second]),
            "research": ["zscheischler2020typology", "ipccar6wg2"],
        })
    return out


def _detect_preconditioned(signals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Antecedent conditions that plausibly amplify a current hazard — always
    labelled INFERRED with the mechanism declared."""

    out: List[Dict[str, Any]] = []
    drought = signals.get("drought") or {}
    fire = signals.get("wildfire") or {}
    heat = signals.get("heat") or {}

    dvals = drought.get("values") or {}
    fvals = fire.get("values") or {}
    hvals = heat.get("values") or {}

    # 90-day precipitation deficit amplifying current fire danger.
    p90 = dvals.get("precipitation_90d") or {}
    deficit_mm = p90.get("deficit_mm")
    fwi_latest = fvals.get("fwi_latest")
    if (deficit_mm is not None and deficit_mm > 0
            and fwi_latest is not None and fwi_latest >= _FWI_ELEVATED):
        out.append({
            "type": "preconditioned",
            "hazards": ["drought", "wildfire"],
            "values": {
                "precipitation_deficit_90d_mm": deficit_mm,
                "deficit_period": p90.get("period"),
                "deficit_standardized_anomaly": p90.get("standardized_anomaly"),
                "fwi_latest": fwi_latest,
                "fwi_latest_date": fvals.get("fwi_latest_date"),
            },
            "claim_status": ClaimStatus.INFERRED.value,
            "temporal": TemporalClass.OBSERVED.value,
            "mechanism": (
                f"Antecedent 90-day precipitation deficit of {deficit_mm} mm "
                f"({p90.get('period', {}).get('start')} to "
                f"{p90.get('period', {}).get('end')}) plausibly amplifies the "
                f"current fire danger (FWI {fwi_latest} >= {_FWI_ELEVATED:.0f}): "
                "sustained deficit dries fuels, raising fire danger for a "
                "given weather day."
            ),
            "basis": "Preconditioned compound signal (zscheischler2020typology): "
                     "an antecedent condition plausibly amplifying a current "
                     "hazard. INFERRED — the amplification pathway is declared, "
                     "not measured.",
            "evidence": _merged_evidence(signals, ["drought", "wildfire"]),
            "research": ["zscheischler2020typology", "ipccar6wg2"],
        })

    # Soil-moisture deficit amplifying an ongoing heat spell.
    soil = dvals.get("soil_moisture") or {}
    soil_anom = soil.get("anomaly_m3m3")
    heat_ongoing = bool(hvals.get("spell_ongoing"))
    if soil_anom is not None and soil_anom < 0 and heat_ongoing:
        out.append({
            "type": "preconditioned",
            "hazards": ["drought", "heat"],
            "values": {
                "soil_moisture_anomaly_m3m3": soil_anom,
                "soil_moisture_percentile": soil.get("percentile_vs_climatology"),
                "soil_moisture_as_of": soil.get("as_of"),
                "heat_spell_ongoing": True,
                "heat_latest": hvals.get("latest"),
            },
            "claim_status": ClaimStatus.INFERRED.value,
            "temporal": TemporalClass.OBSERVED.value,
            "mechanism": (
                f"Soil-moisture deficit ({soil_anom} m³/m³ vs the day-of-year "
                "climatology) during an ongoing heat spell plausibly amplifies "
                "it: dry soils reduce evaporative cooling, so more incoming "
                "energy heats the surface and near-surface air "
                "(land–atmosphere coupling)."
            ),
            "basis": "Preconditioned compound signal (zscheischler2020typology): "
                     "an antecedent condition plausibly amplifying a current "
                     "hazard. INFERRED — the amplification pathway is declared, "
                     "not measured.",
            "evidence": _merged_evidence(signals, ["drought", "heat"]),
            "research": ["zscheischler2020typology", "ipccar6wg2"],
        })
    return out


# ---------------------------------------------------------------------------
# Empirical dependence analysis (additive — descriptive co-occurrence only)
# ---------------------------------------------------------------------------

_DEPENDENCE_WINDOW_YEARS = 10    # trailing record length for the analysis
_DEPENDENCE_MIN_EVENT_DAYS = 10  # small-count guard: fewer -> insufficient_data
_DEPENDENCE_DOY_POOL_DAYS = 7    # day-of-year climatology pool half-width (heat)
_DEPENDENCE_DROUGHT_ROLL_DAYS = 30  # rolling precipitation window (drought z)

#: Elevated-indicator pairs analysed (the three indicators computable from
#: the location's own daily ERA5-derived series).
_DEPENDENCE_PAIRS = (
    ("heat", "drought"),
    ("heat", "wildfire"),
    ("drought", "wildfire"),
)

_DEPENDENCE_INDICATORS = {
    "heat": (
        "event day: daily Tmax >= the 90th percentile of the day-of-year "
        "climatology (±7-day pool over the record's other years), consistent "
        "with the declared heat elevated threshold"
    ),
    "drought": (
        "event day: trailing 30-day precipitation-sum standardized anomaly "
        "z <= -1.0 vs the same calendar window in the record's other years "
        "(declared; NOT a fitted SPI/SPEI)"
    ),
    "wildfire": (
        "event day: daily Canadian FWI >= 30 (declared fire-danger "
        "screening threshold)"
    ),
}

_DEPENDENCE_METHOD = (
    "empirical conditional frequencies from the location's own {years}-year "
    "daily series; NOT a fitted dependence model; no causal claim"
)

_DEPENDENCE_SMALL_COUNT_GUARD = (
    "If either event count is < 10 days the pair is reported as "
    "insufficient_data and no frequencies or lift are computed (declared "
    "small-count guard)."
)

_DEPENDENCE_SIGNIFICANCE_NOTE = (
    "No significance testing is performed: daily samples are serially "
    "correlated and no declared exact method is implemented here, so "
    "p-values would be pseudo-precise. The lift is a raw empirical ratio "
    "only."
)

_DEPENDENCE_LIMITATIONS = [
    "Co-occurrence frequencies are descriptive; correlation is not "
    "causation and no driver attribution is made.",
    _DEPENDENCE_SIGNIFICANCE_NOTE,
    "Serial correlation between adjacent days is not modelled; the "
    "effective sample size is smaller than n_days_total.",
    "Events are declared exceedances of screening thresholds on reanalysis "
    "series (ERA5, ~25 km grid), not observed impact events.",
    "FWI screening approximation (daily aggregates, not noon-standard "
    "inputs); FWI codes seeded with conventional startup values at the "
    "window start.",
    "Leap-year day numbering in the day-of-year pools is a declared "
    "approximation (±1 day at the year boundary).",
]


def _valid_daily_pairs(dates: Any, values: Any) -> List[Tuple[date, float]]:
    """Parsed, sorted (date, value) pairs; unparseable/None entries dropped."""

    out: List[Tuple[date, float]] = []
    for d, v in zip(dates or [], values or []):
        dd = _series.parse_date(d)
        if dd is None or v is None:
            continue
        try:
            out.append((dd, float(v)))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda dv: dv[0])
    return out


def _heat_event_days(days: List[Tuple[date, float]]) -> Dict[str, bool]:
    """Event = Tmax >= p90 of the day-of-year pool (other years, ±7 days).

    Same declared statistic as _series.doy_thresholds; pools are gathered
    from pre-bucketed day-of-year lists so a 10-year record stays cheap.
    """

    by_doy: Dict[int, List[Tuple[int, float]]] = {}
    for d, v in days:
        by_doy.setdefault(d.timetuple().tm_yday, []).append((d.year, v))
    flags: Dict[str, bool] = {}
    for d, v in days:
        doy = d.timetuple().tm_yday
        pool: List[float] = []
        for delta in range(-_DEPENDENCE_DOY_POOL_DAYS, _DEPENDENCE_DOY_POOL_DAYS + 1):
            dd = ((doy - 1 + delta) % 365) + 1
            pool.extend(val for yr, val in by_doy.get(dd, ()) if yr != d.year)
        thr = _series.percentile_value(pool, _PCT_ELEVATED)
        flags[d.isoformat()] = bool(thr is not None and v >= thr)
    return flags


def _drought_event_days(days: List[Tuple[date, float]]) -> Dict[str, bool]:
    """Event = trailing 30-day precipitation-sum z <= -1 vs other years.

    The baseline for each day is the same calendar window (month-day) in
    the record's other years; z via _series.standardized_anomaly (None —
    no event — where the baseline is too small or has zero variance).
    """

    dates = [d for d, _ in days]
    sums = _series.rolling_sums([v for _, v in days], _DEPENDENCE_DROUGHT_ROLL_DAYS)
    by_calday: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for d, s in zip(dates, sums):
        if s is not None:
            by_calday.setdefault((d.month, d.day), []).append((d.year, s))
    flags: Dict[str, bool] = {}
    for d, s in zip(dates, sums):
        if s is None:
            flags[d.isoformat()] = False
            continue
        baseline = [v for yr, v in by_calday.get((d.month, d.day), ()) if yr != d.year]
        z, _, _ = _series.standardized_anomaly(s, baseline)
        flags[d.isoformat()] = bool(z is not None and z <= _DROUGHT_Z_ELEVATED)
    return flags


def _fwi_event_days(days: List[Tuple[date, float]]) -> Dict[str, bool]:
    """Event = daily FWI >= the declared fire-danger threshold."""

    return {d.isoformat(): bool(v >= _FWI_ELEVATED) for d, v in days}


_EVENT_BUILDERS = {
    "heat": _heat_event_days,
    "drought": _drought_event_days,
    "wildfire": _fwi_event_days,
}


def dependence_analysis(
    series_by_hazard: Dict[str, Any],
    *,
    window_years: int = _DEPENDENCE_WINDOW_YEARS,
) -> Dict[str, Any]:
    """Empirical co-occurrence of the elevated indicators over the record.

    ``series_by_hazard`` maps "heat" / "drought" / "wildfire" to
    ``{"dates": [...], "values": [...]}`` daily series from the location's
    own ERA5-derived record (heat: daily Tmax °C; drought: daily
    precipitation mm; wildfire: daily FWI). For each declared pair the
    trailing ``window_years`` record yields P(A), P(B), P(A∩B) and the
    empirical lift P(A∩B)/(P(A)·P(B)) — real counts only. Below 10 event
    days on either side the pair is ``insufficient_data`` and carries no
    frequencies or lift. Pure function: no I/O, no fitted model, no
    significance testing, no causal claim.
    """

    window_days = int(round(window_years * 365.25))
    event_days: Dict[str, Dict[str, bool]] = {}
    series_windows: Dict[str, Any] = {}
    unavailable_hazards: List[str] = []

    for hid in ("heat", "drought", "wildfire"):
        raw = (series_by_hazard or {}).get(hid) or {}
        parsed = _valid_daily_pairs(raw.get("dates"), raw.get("values"))
        if len(parsed) < 2:
            unavailable_hazards.append(hid)
            continue
        end = parsed[-1][0]
        start = end - timedelta(days=window_days)
        in_window = [(d, v) for d, v in parsed if d >= start]
        series_windows[hid] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_days": len(in_window),
        }
        event_days[hid] = _EVENT_BUILDERS[hid](in_window)

    pairs: Dict[str, Any] = {}
    n_pairs_ok = 0
    for a, b in _DEPENDENCE_PAIRS:
        key = f"{a}_{b}"
        entry: Dict[str, Any] = {
            "hazards": [a, b],
            "indicator_a": _DEPENDENCE_INDICATORS[a],
            "indicator_b": _DEPENDENCE_INDICATORS[b],
            "n_days_total": None,
            "n_A": None,
            "n_B": None,
            "n_AB": None,
            "P_A": None,
            "P_B": None,
            "P_AB": None,
            "lift": None,
        }
        fa, fb = event_days.get(a), event_days.get(b)
        if fa is None or fb is None:
            missing = [h for h in (a, b) if event_days.get(h) is None]
            entry["status"] = "unavailable"
            entry["reason"] = ("daily series unavailable for: "
                               + ", ".join(missing))
            pairs[key] = entry
            continue
        common = sorted(set(fa) & set(fb))
        n = len(common)
        n_a = sum(1 for d in common if fa[d])
        n_b = sum(1 for d in common if fb[d])
        n_ab = sum(1 for d in common if fa[d] and fb[d])
        entry.update(n_days_total=n, n_A=n_a, n_B=n_b, n_AB=n_ab)
        if n < 1 or n_a < _DEPENDENCE_MIN_EVENT_DAYS or n_b < _DEPENDENCE_MIN_EVENT_DAYS:
            entry["status"] = "insufficient_data"
            entry["reason"] = (
                f"small-count guard: event days n_A={n_a}, n_B={n_b} over "
                f"{n} common days; at least {_DEPENDENCE_MIN_EVENT_DAYS} "
                "event days on each side are required before empirical "
                "frequencies are reported."
            )
        else:
            p_a = n_a / n
            p_b = n_b / n
            p_ab = n_ab / n
            entry.update(
                status="ok",
                P_A=round(p_a, 4),
                P_B=round(p_b, 4),
                P_AB=round(p_ab, 4),
                lift=round(p_ab / (p_a * p_b), 3),
            )
            n_pairs_ok += 1
        pairs[key] = entry

    if n_pairs_ok == len(_DEPENDENCE_PAIRS):
        status = "ok"
    elif n_pairs_ok > 0:
        status = "partial"
    else:
        status = "unavailable"

    return {
        "status": status,
        "window_years": window_years,
        "series_windows": series_windows,
        "series_unavailable": unavailable_hazards,
        "indicators": dict(_DEPENDENCE_INDICATORS),
        "pairs": pairs,
        "claim_status": ClaimStatus.MODELLED.value,
        "method": _DEPENDENCE_METHOD.format(years=window_years),
        "small_count_guard": _DEPENDENCE_SMALL_COUNT_GUARD,
        "significance_note": _DEPENDENCE_SIGNIFICANCE_NOTE,
        "limitations": list(_DEPENDENCE_LIMITATIONS),
    }


def _dependence_unavailable(reason: str) -> Dict[str, Any]:
    """Honest dependence block when the series cannot be fetched/analysed."""

    return {
        "status": "unavailable",
        "reason": reason,
        "window_years": _DEPENDENCE_WINDOW_YEARS,
        "series_windows": {},
        "series_unavailable": [],
        "indicators": dict(_DEPENDENCE_INDICATORS),
        "pairs": {},
        "claim_status": ClaimStatus.MODELLED.value,
        "method": _DEPENDENCE_METHOD.format(years=_DEPENDENCE_WINDOW_YEARS),
        "small_count_guard": _DEPENDENCE_SMALL_COUNT_GUARD,
        "significance_note": _DEPENDENCE_SIGNIFICANCE_NOTE,
        "limitations": list(_DEPENDENCE_LIMITATIONS),
    }


@cached("compound_dependence_series", TTL_COMPOUND)
def _dependence_series(
    lat: float,
    lon: float,
    window_years: int = _DEPENDENCE_WINDOW_YEARS,
) -> Dict[str, Any]:
    """Trailing ``window_years`` daily series for the dependence analysis.

    The same ERA5 daily archive fetcher the light signals use
    (``real_data.fetch_weather_archive``), over the trailing record; FWI is
    computed with the same declared screening approximation as
    ``_fire_danger_signal``. Cached 1 h.
    """

    from ..dashboard import real_data as rd
    from ..prediction.fwi import compute_fwi_series

    today = date.today()
    end = today - timedelta(days=_ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=int(round(window_years * 365.25)))
    try:
        archive = rd.fetch_weather_archive(
            lat, lon, start.isoformat(), end.isoformat())
    except Exception as exc:
        return {"error": f"ERA5 dependence-series fetch failed: {exc}"}
    if "error" in archive:
        return {"error": archive["error"]}

    times = archive.get("time") or []
    tmax = archive.get("temperature_2m_max") or []
    rh = archive.get("relative_humidity_2m_mean") or []
    wind = archive.get("wind_speed_10m_max") or []
    rain = archive.get("precipitation_sum") or []
    heat_dates: List[str] = []
    heat_vals: List[float] = []
    pr_dates: List[str] = []
    pr_vals: List[float] = []
    series_in: List[Dict[str, Any]] = []
    for i, t in enumerate(times):
        if i < len(tmax) and tmax[i] is not None:
            heat_dates.append(t)
            heat_vals.append(float(tmax[i]))
        if i < len(rain) and rain[i] is not None:
            pr_dates.append(t)
            pr_vals.append(float(rain[i]))
        if i >= len(tmax) or i >= len(rh) or i >= len(wind):
            continue
        if tmax[i] is None or rh[i] is None or wind[i] is None:
            continue
        series_in.append({
            "date": t,
            "temp_c": float(tmax[i]),
            "rh_pct": float(rh[i]),
            "wind_kmh": float(wind[i]),
            "rain_mm": float(rain[i]) if i < len(rain) and rain[i] is not None else 0.0,
        })
    fwi_dates: List[str] = []
    fwi_vals: List[float] = []
    if len(series_in) >= 5:
        for day in compute_fwi_series(series_in):
            fwi_dates.append(str(day.date))
            fwi_vals.append(round(day.fwi, 1))

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "series_by_hazard": {
            "heat": {"dates": heat_dates, "values": heat_vals},
            "drought": {"dates": pr_dates, "values": pr_vals},
            "wildfire": {"dates": fwi_dates, "values": fwi_vals},
        },
        "source": archive.get("source", "Reanalysis (ERA5 via Open-Meteo archive)"),
    }


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@cached("compound_assess", TTL_COMPOUND)
def assess_compound(
    lat: float,
    lon: float,
    *,
    hazards: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compound-risk assessment for a point. Cached 1 h.

    Returns a qualitative signal list — never a numeric compound score.
    """

    lat, lon = float(lat), float(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Coordinates out of range"}

    extracted = extract_light_signals(lat, lon, hazards=list(hazards) if hazards else None)
    signals: Dict[str, Any] = extracted["signals"]
    unavailable: List[Dict[str, str]] = extracted["hazards_unavailable"]

    compound_signals: List[Dict[str, Any]] = []
    compound_signals.extend(_detect_multivariate(signals))
    compound_signals.extend(_detect_temporal(signals, extracted.get("as_of")))
    compound_signals.extend(_detect_preconditioned(signals))

    # Empirical dependence analysis (additive): co-occurrence of the declared
    # elevated indicators over the location's own trailing daily series.
    # Descriptive only; failures degrade to an honest unavailable block and
    # never affect the qualitative signals above.
    try:
        dep_series = _dependence_series(lat, lon)
    except Exception as exc:  # pragma: no cover - defensive
        dep_series = {"error": f"dependence series fetch failed: {exc}"}
    if "error" in dep_series:
        dependence = _dependence_unavailable(str(dep_series["error"]))
    else:
        try:
            dependence = dependence_analysis(dep_series.get("series_by_hazard") or {})
            dependence["source"] = dep_series.get("source")
        except Exception as exc:  # pragma: no cover - defensive
            dependence = _dependence_unavailable(f"dependence analysis failed: {exc}")

    analysed = sorted(set(signals) | {u["hazard"] for u in unavailable})
    if not signals:
        status = "unavailable"
    elif unavailable:
        status = "partial"
    else:
        status = "ok"

    per_hazard_sources = {
        hid: sig.get("source") for hid, sig in sorted(signals.items())
    }

    return {
        "status": status,
        "location": {"lat": lat, "lon": lon},
        "generated_at": extracted["generated_at"],
        "as_of": extracted.get("as_of"),
        "hazards_analysed": analysed,
        "hazards_unavailable": unavailable,
        "hazard_signals": signals,
        "thresholds": extracted["thresholds"],
        "compound_signals": compound_signals,
        "dependence": dependence,
        "no_compound_signal": None if compound_signals else {
            "status": "no_compound_signal",
            "statement": _NO_SIGNAL_STATEMENT,
        },
        "spatially_compounding": {
            "status": "not_computable",
            "reason": _SPATIALLY_NOT_COMPUTABLE_REASON,
        },
        "uncertainty": {
            "confidence": Confidence.LOW.value,
            "note": "Screening-level qualitative detection over reanalysis-based "
                    "hazard signals; not a validated compound-event predictor.",
            "sources_of_uncertainty": [
                "ERA5/ERA5-Land reanalysis (~11-25 km grid), not station measurements",
                f"Open-Meteo archive lag ~{_ARCHIVE_LAG_DAYS} days behind real time",
                "Declared screening thresholds; no calibrated joint-probability model",
                "FWI screening approximation (daily aggregates; northern-hemisphere "
                "day-length tables)",
            ],
        },
        "limitations": [
            _NO_NUMERIC_SCORE_NOTE,
            _SPATIALLY_NOT_COMPUTABLE_REASON,
            "Fire danger is not fire occurrence; high FWI means conditions "
            "favour spread IF an ignition happens.",
            "Preconditioned signals are INFERRED: the amplification mechanism "
            "is declared, not measured.",
            "The dependence block reports empirical co-occurrence frequencies "
            "over the location's own trailing record — descriptive only, NOT "
            "a fitted dependence model; no causal claim, no significance "
            "testing.",
        ],
        "provenance": {
            "engine": "compound_risk_v1",
            "typology_basis": _TYPOLOGY_BASIS,
            "research": [
                {"id": "zscheischler2020typology",
                 "role": "four-class compound-event typology (declared detector on top)"},
                {"id": "ipccar6wg2",
                 "role": "risk framework: hazard x exposure x vulnerability; "
                         "compound/cascading risk"},
            ],
            "per_hazard_sources": per_hazard_sources,
            "no_numeric_score": _NO_NUMERIC_SCORE_NOTE,
        },
    }
