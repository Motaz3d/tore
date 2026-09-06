"""
Drought hazard plugin (Stage 4 foundation).

Real-data analyses only (ERA5 / ERA5-Land via the Open-Meteo archive):

- **Precipitation deficit** — accumulated deficit vs the same calendar
  window in previous years over 30/90/180-day windows. Declared method:
  standardized anomaly (z = (x − mean) / sample std of the baseline years).
  NOT a full SPEI — no distribution fitting is performed or claimed.
- **Soil moisture** — ERA5-Land 0–7 cm soil moisture anomaly vs the
  day-of-year climatology of the same grid point.
- **Atmospheric demand** — FAO ET₀ vs precipitation balance over 90 days.
- **Severity / duration / trend** — dry-spell detection on the daily series
  (declared threshold), rank of the current window within previous years.
- **Agricultural exposure** — ESA WorldCover cropland fraction (observed
  land cover) around the point.

Everything computed by Talaix is labelled MODELLED with its method.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from ..ontology import ClaimStatus, EvidenceClass, TemporalClass
from ..evidence import EvidenceRecord, content_hash
from .base import HazardAnalysis, HazardLevel, HazardModule, LayerSpec
from . import _series

_HISTORY_DAYS = 365 * 10
_ARCHIVE_LAG_DAYS = 5
_WINDOWS = (30, 90, 180)
_DRY_THRESHOLD_MM = 1.0       # a "dry day" has < 1 mm precipitation (declared)
_DRY_MIN_DAYS = 10            # >= 10 consecutive dry days = one dry spell
_SM_CURRENT_DAYS = 7          # current soil moisture = mean of last 7 valid days
_BALANCE_WINDOW = 90

_VARIABLES = (
    "precipitation_sum",
    "soil_moisture_0_to_7cm_mean",
    "et0_fao_evapotranspiration",
)


class DroughtModule(HazardModule):
    id = "drought"
    name = "Drought"
    tagline = "Precipitation deficit, soil-moisture anomaly, water balance and agricultural exposure."

    def temporal_coverage(self) -> Dict[str, Dict[str, str]]:
        return {
            "ERA5 / ERA5-Land daily (Open-Meteo archive)": {
                "start": "1940",
                "end": "~5 days ago (archive lag)",
            },
            "ESA WorldCover (agricultural exposure)": {"start": "2021", "end": "2021"},
        }

    # -- analysis ------------------------------------------------------

    def analyze(self, lat: float, lon: float, name: Optional[str] = None, **kw: Any) -> HazardAnalysis:
        from ...dashboard import real_data as rd
        from ...gis_mapping.landcover import fetch_landcover

        today = date.today()
        hist_start = (today - timedelta(days=_HISTORY_DAYS)).isoformat()
        archive_end = (today - timedelta(days=_ARCHIVE_LAG_DAYS)).isoformat()
        location = {"lat": lat, "lon": lon}

        climate = rd.fetch_daily_climate(lat, lon, hist_start, archive_end, list(_VARIABLES))
        landcover = fetch_landcover(lat, lon)

        blocks: Dict[str, Any] = {}
        evidence: List[Dict[str, Any]] = []
        provenance: Dict[str, Any] = {}

        if "error" in climate:
            reason = climate["error"]
            blocks["precipitation_deficit"] = {"status": "unavailable", "reason": reason}
            blocks["agricultural_exposure"] = self._landcover_block(landcover)
            rec = EvidenceRecord.unknown("Open-Meteo archive (ERA5/ERA5-Land)", why=reason)
            evidence.append(rec.to_dict())
            provenance["climate"] = rec.to_dict()
            if blocks["agricultural_exposure"].get("status") == "ok":
                lrec = self._landcover_record(landcover, lat, lon)
                evidence.append(lrec.to_dict())
                provenance["agricultural_exposure"] = lrec.to_dict()
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Drought analysis unavailable for this location.",
                blocks=blocks,
                evidence=evidence,
                provenance=provenance,
                unavailable_reason=reason,
            )

        times: List[str] = climate["time"]
        pr: List[Optional[float]] = climate["precipitation_sum"]
        sm: List[Optional[float]] = climate["soil_moisture_0_to_7cm_mean"]
        et0: List[Optional[float]] = climate["et0_fao_evapotranspiration"]
        unavailable_vars = set(climate.get("unavailable_variables") or [])

        pr_ok = any(v is not None for v in pr)
        sm_ok = "soil_moisture_0_to_7cm_mean" not in unavailable_vars and any(
            v is not None for v in sm
        )
        et0_ok = "et0_fao_evapotranspiration" not in unavailable_vars and any(
            v is not None for v in et0
        )

        if not pr_ok:
            reason = "Archive returned no usable precipitation data for this location."
            blocks["precipitation_deficit"] = {"status": "unavailable", "reason": reason}
            rec = EvidenceRecord.unknown("Open-Meteo archive (ERA5)", why=reason)
            evidence.append(rec.to_dict())
            provenance["climate"] = rec.to_dict()
            return HazardAnalysis(
                hazard=self.id,
                location=location,
                status="unavailable",
                summary="Drought analysis unavailable for this location.",
                blocks=blocks,
                evidence=evidence,
                provenance=provenance,
                unavailable_reason=reason,
            )

        # -- precipitation deficit (30/90/180-day windows) --------------
        deficit_block, min_z, trend = self._deficit_block(times, pr)
        blocks["precipitation_deficit"] = deficit_block

        # -- soil moisture anomaly ---------------------------------------
        blocks["soil_moisture"] = (
            self._soil_block(times, sm) if sm_ok else {
                "status": "unavailable",
                "reason": "Soil moisture not exposed by the archive for this location "
                          "(e.g. over water). Not estimated.",
            }
        )

        # -- ET0 vs precipitation balance --------------------------------
        blocks["water_balance"] = (
            self._balance_block(times, pr, et0) if (pr_ok and et0_ok) else {
                "status": "unavailable",
                "reason": "ET₀ not exposed by the archive for this location; "
                          "water balance not computed.",
            }
        )

        # -- dry spells ----------------------------------------------------
        blocks["dry_spells"] = self._dry_spell_block(times, pr)

        # -- agricultural exposure -----------------------------------------
        blocks["agricultural_exposure"] = self._landcover_block(landcover)

        # -- evidence ------------------------------------------------------
        rec = EvidenceRecord(
            EvidenceClass.OPEN_DATA_OFFICIAL.value,
            ClaimStatus.MODELLED.value,
            TemporalClass.HISTORICAL.value,
            climate["source"],
            dataset="ERA5 daily precipitation; ERA5-Land soil moisture 0–7 cm; FAO ET₀",
            provider_url="https://open-meteo.com/en/docs/historical-weather-api",
            link=climate.get("request_url"),
            location={"lat": lat, "lon": lon},
            reference_period={"start": hist_start, "end": archive_end},
            method=(
                "Precipitation deficit = standardized anomaly of 30/90/180-day sums vs "
                "the same calendar windows of previous years (sample std, n−1; baseline "
                "needs >= 5 years). NOT a full SPEI (no distribution fitting). Soil "
                "moisture anomaly vs ±7-day day-of-year climatology. Archive lag "
                f"~{_ARCHIVE_LAG_DAYS} days."
            ),
            resolution="~11 km (ERA5-Land) / ~25 km (ERA5)",
            limitations="Reanalysis, not station measurements; soil moisture is modelled.",
            content_hash=content_hash(
                {"time": times, "precipitation_sum": pr,
                 "soil_moisture_0_to_7cm_mean": sm, "et0_fao_evapotranspiration": et0}
            ),
        )
        evidence.append(rec.to_dict())
        provenance["climate"] = rec.to_dict()
        if blocks["agricultural_exposure"].get("status") == "ok":
            lrec = self._landcover_record(landcover, lat, lon)
            evidence.append(lrec.to_dict())
            provenance["agricultural_exposure"] = lrec.to_dict()

        # -- level + status ------------------------------------------------
        level = self._level(min_z)
        status = "ok" if (sm_ok and et0_ok) else "partial"
        summary = self._summary(deficit_block, trend, level, status)
        return HazardAnalysis(
            hazard=self.id,
            location=location,
            status=status,
            summary=summary,
            level=level,
            blocks=blocks,
            evidence=evidence,
            provenance=provenance,
        )

    # -- blocks ----------------------------------------------------------

    def _deficit_block(self, times, pr):
        """30/90/180-day deficits; returns (block, min z, trend dict)."""

        windows: Dict[str, Any] = {}
        min_z: Optional[float] = None
        trend: Dict[str, Any] = {}
        for w in _WINDOWS:
            yearly = _series.window_sums_by_year(times, pr, w, years_back=10)
            if len(yearly) < 2:
                windows[str(w)] = {
                    "status": "unavailable",
                    "reason": f"Fewer than 2 complete {w}-day windows in the series.",
                }
                continue
            current = yearly[0]
            baseline = [y["sum"] for y in yearly[1:]]
            z, mean, std = _series.standardized_anomaly(current["sum"], baseline)
            if z is not None and (min_z is None or z < min_z):
                min_z = z
            lower = sum(1 for b in baseline if b < current["sum"])
            windows[str(w)] = {
                "status": "ok",
                "claim_status": ClaimStatus.MODELLED.value,
                "window_days": w,
                "current_period": {"start": current["start"], "end": current["end"]},
                "current_sum_mm": current["sum"],
                "climatology_mean_mm": round(mean, 2) if mean is not None else None,
                "deficit_mm": round(mean - current["sum"], 2) if mean is not None else None,
                "standardized_anomaly": round(z, 2) if z is not None else None,
                "baseline_years": len(baseline),
                "years_with_less_precipitation": lower,
                "per_year_sums_mm": yearly,
            }
            if w == 90:
                trend = {
                    "rank_driest": lower + 1,
                    "years_in_comparison": len(yearly),
                    "note": (
                        f"The current 90-day window is the {self._ordinal(lower + 1)} "
                        f"driest of the last {len(yearly)} same-calendar windows."
                    ),
                }
        block = {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "windows": windows,
            "method": (
                "Standardized anomaly z = (current W-day sum − mean of same-calendar "
                "W-day sums of previous years) / sample std (n−1) of those years; "
                "baseline = up to 10 previous years, needs ≥ 5 for z. "
                "NOT a full SPEI — no distribution fitting is performed."
            ),
        }
        return block, min_z, trend

    @staticmethod
    def _soil_block(times, sm):
        valid = [(t, v) for t, v in zip(times, sm) if v is not None]
        if not valid:
            return {"status": "unavailable", "reason": "No usable soil-moisture values."}
        last_date = valid[-1][0]
        current_vals = [v for _t, v in valid[-_SM_CURRENT_DAYS:]]
        current = sum(current_vals) / len(current_vals)
        pool = _series.doy_window_pool(times, sm, last_date, window_days=7)
        if len(pool) < 30:
            return {
                "status": "unavailable",
                "reason": "Climatology pool too small for an honest anomaly.",
            }
        mean = sum(pool) / len(pool)
        return {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "current_m3m3": round(current, 3),
            "as_of": last_date,
            "climatology_mean_m3m3": round(mean, 3),
            "anomaly_m3m3": round(current - mean, 3),
            "percentile_vs_climatology": _series.percentile_rank(pool, current),
            "method": (
                f"Current = mean of last {_SM_CURRENT_DAYS} valid days; climatology = "
                f"±7-day day-of-year pool across all other years of the series "
                f"({len(pool)} values). Dataset: ERA5-Land soil moisture 0–7 cm."
            ),
        }

    @staticmethod
    def _balance_block(times, pr, et0):
        pairs = [
            (t, p, e) for t, p, e in zip(times, pr, et0)
            if p is not None and e is not None
        ]
        if len(pairs) < _BALANCE_WINDOW:
            return {"status": "unavailable", "reason": "Insufficient overlapping P/ET₀ data."}
        window = pairs[-_BALANCE_WINDOW:]
        p_sum = sum(p for _t, p, _e in window)
        e_sum = sum(e for _t, _p, e in window)
        return {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "window": {"start": window[0][0], "end": window[-1][0]},
            "precipitation_sum_mm": round(p_sum, 1),
            "et0_sum_mm": round(e_sum, 1),
            "balance_mm": round(p_sum - e_sum, 1),
            "method": (
                f"Climatic water balance = Σ precipitation − Σ FAO ET₀ over the last "
                f"{_BALANCE_WINDOW} days with both variables present. A simple "
                f"screening balance — not a soil-water model."
            ),
        }

    @staticmethod
    def _dry_spell_block(times, pr):
        valid = [(t, v) for t, v in zip(times, pr) if v is not None]
        last365 = valid[-365:]
        spells = _series.detect_spells(
            [t for t, _v in last365],
            [v for _t, v in last365],
            _DRY_THRESHOLD_MM,
            min_len=_DRY_MIN_DAYS,
            above=False,
        )
        longest = max(spells, key=lambda s: s["length_days"], default=None)
        ongoing = bool(spells and spells[-1]["end"] == last365[-1][0])
        return {
            "status": "ok",
            "claim_status": ClaimStatus.MODELLED.value,
            "temporal": TemporalClass.HISTORICAL.value,
            "spells_last_year": spells,
            "count_last_year": len(spells),
            "longest_days": longest["length_days"] if longest else 0,
            "ongoing": ongoing,
            "method": (
                f"Dry spell = ≥{_DRY_MIN_DAYS} consecutive days with daily precipitation "
                f"< {_DRY_THRESHOLD_MM} mm (declared threshold), over the last 365 days."
            ),
        }

    @staticmethod
    def _landcover_block(landcover):
        if "error" in landcover:
            return {"status": "unavailable", "reason": landcover["error"]}
        hist = landcover.get("histogram") or {}
        cropland = hist.get(40) or hist.get("40") or {}
        return {
            "status": "ok",
            "claim_status": ClaimStatus.OBSERVED.value,
            "cropland_fraction": cropland.get("fraction", 0.0),
            "dominant_landcover": landcover.get("dominant_label"),
            "resolution": landcover.get("resolution"),
            "note": "ESA WorldCover cropland class share in the analysis window.",
            "source": landcover.get("source"),
        }

    @staticmethod
    def _landcover_record(landcover, lat, lon):
        return EvidenceRecord.open_data(
            landcover["source"],
            status=ClaimStatus.OBSERVED.value,
            temporal=TemporalClass.OBSERVED.value,
            dataset="ESA WorldCover 10 m 2021 v200",
            location={"lat": lat, "lon": lon},
            method="Cropland class (40) fraction of the WorldCover histogram window.",
            resolution=landcover.get("resolution"),
            limitations="Classified land-cover product; 2021 snapshot.",
        )

    # -- level / summary ---------------------------------------------------

    @staticmethod
    def _level(min_z: Optional[float]) -> Optional[HazardLevel]:
        if min_z is None:
            return None
        if min_z <= -2.0:
            label = "Extreme"
        elif min_z <= -1.5:
            label = "Severe"
        elif min_z <= -1.0:
            label = "Moderate"
        elif min_z <= -0.5:
            label = "Mild"
        else:
            label = "Near normal"
        return HazardLevel(
            label=label,
            score=round(min_z, 2),
            basis=(
                "Screening indicator: worst standardized precipitation anomaly across "
                "the 30/90/180-day windows, banded ≤−2 Extreme / ≤−1.5 Severe / "
                "≤−1 Moderate / ≤−0.5 Mild / else Near normal (anomaly classes in the "
                "spirit of WMO drought monitoring; NOT a validated predictor)."
            ),
            validated=False,
        )

    @staticmethod
    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _summary(self, deficit_block, trend, level, status) -> str:
        parts: List[str] = []
        w90 = (deficit_block.get("windows") or {}).get("90") or {}
        if w90.get("status") == "ok":
            parts.append(
                f"90-day precipitation {w90['current_sum_mm']:.0f} mm "
                f"(deficit {w90.get('deficit_mm')} mm, z={w90.get('standardized_anomaly')})"
            )
        if trend:
            parts.append(trend["note"])
        if level is not None:
            parts.append(f"screening level: {level.label}")
        if status == "partial":
            parts.append("partial: soil moisture and/or ET₀ unavailable here")
        return "; ".join(parts) + "." if parts else "Drought analysis complete."

    # -- map layers --------------------------------------------------------

    def map_layers(self, **kw: Any) -> list:
        return [
            LayerSpec(
                layer_id="drought.deficit",
                label="Precipitation deficit (standardized anomaly)",
                group="HAZARD",
                kind="grid",
                endpoint="/api/v2/analyze?hazard=drought&lat={lat}&lon={lon}",
                legend={"Near normal": "#22c55e", "Mild": "#eab308", "Moderate": "#f97316",
                        "Severe": "#ef4444", "Extreme": "#7f1d1d"},
                source="ERA5 daily precipitation (Open-Meteo archive); declared anomaly method",
                url="https://open-meteo.com/en/docs/historical-weather-api",
                resolution="~25 km reanalysis grid",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
                default_on=True,
            ).to_dict(),
            LayerSpec(
                layer_id="drought.soil_moisture",
                label="Soil-moisture anomaly (ERA5-Land 0–7 cm)",
                group="ENVIRONMENT",
                kind="grid",
                endpoint="/api/v2/analyze?hazard=drought&lat={lat}&lon={lon}",
                source="ERA5-Land soil moisture (Open-Meteo archive)",
                url="https://open-meteo.com/en/docs/historical-weather-api",
                resolution="~11 km",
                status="available",
                temporal=TemporalClass.HISTORICAL.value,
            ).to_dict(),
            LayerSpec(
                layer_id="drought.cropland",
                label="Cropland exposure (ESA WorldCover)",
                group="EXPOSURE",
                kind="raster",
                source="ESA WorldCover 10 m 2021 v200",
                url="https://esa-worldcover.org/",
                resolution="10 m",
                status="available",
                temporal=TemporalClass.OBSERVED.value,
            ).to_dict(),
        ]
