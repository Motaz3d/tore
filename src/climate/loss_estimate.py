"""
Talaix Loss Screening Estimate — the ESTIMATED monetary layer
(docs/ECONOMIC_INTELLIGENCE.md §9).

The platform's first monetary function of its own. It COMPUTES an
exposed-value screening range from real engine inputs (mapped building
counts, location) and declared published benchmarks
(``config/loss_estimate_benchmarks.json``): replacement cost per country
and a floor-area-per-building assumption. This is engine work, not quoted
content — nothing here is a figure republished from someone else's
database.

Honesty contract (absolute, same weight as the no-fake-money rule):

- Inputs are either real (OSM mapped building counts with their
  completeness caveats) or declared (benchmarks carry basis notes and
  deliberately wide ranges — declared screening assumptions, never
  presented as valuations).
- The output is an exposed-VALUE range — what could be at stake — NOT an
  expected loss. The expected-loss slot (damage ratio × exposed value) is
  reported ``not_available`` until a validated damage model is integrated
  (JRC depth–damage functions for flood are the documented next step).
- ESTIMATED figures are never merged with DOCUMENTED loss figures; the
  two blocks stay strictly separated everywhere they are served.
- Wide ranges are deliberate: the function reports its own sensitivity
  instead of pretending precision.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .evidence import utcnow_iso
from .ontology import Confidence

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "loss_estimate_benchmarks.json"
)

EXPECTED_LOSS_STATEMENT = (
    "No validated damage-ratio model is integrated; the exposed value is not "
    "converted into an expected loss. JRC depth–damage functions (flood) are "
    "the documented next step (docs/ECONOMIC_INTELLIGENCE.md §9)."
)

SEPARATION_NOTE = (
    "ESTIMATED figures are computed by the Talaix screening function from "
    "declared benchmarks; they are never merged with DOCUMENTED loss figures "
    "(docs/ECONOMIC_INTELLIGENCE.md §3)."
)


def load_benchmarks(path: str | None = None) -> Dict[str, Any]:
    cfg_path = path or os.environ.get("HYDRASHIELD_LOSS_BENCHMARKS") or _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_benchmarks(cfg: Dict[str, Any]) -> List[str]:
    """Structural validation; returns a list of problems (empty = valid)."""

    problems: List[str] = []

    def _band(problems_out: List[str], where: str, band: Any) -> None:
        if not isinstance(band, dict):
            problems_out.append(f"{where}: must be an object")
            return
        for field in ("low", "central", "high"):
            if not isinstance(band.get(field), (int, float)):
                problems_out.append(f"{where}: missing numeric {field}")
        if isinstance(band.get("low"), (int, float)) and isinstance(
                band.get("high"), (int, float)) and not band["low"] <= band["central"] <= band["high"]:
            problems_out.append(f"{where}: low <= central <= high violated")
        if not band.get("basis"):
            problems_out.append(f"{where}: missing basis note")

    defaults = cfg.get("defaults") or {}
    _band(problems, "defaults.replacement_cost_per_m2",
          defaults.get("replacement_cost_per_m2"))
    _band(problems, "defaults.floor_area_per_building_m2",
          defaults.get("floor_area_per_building_m2"))

    countries = cfg.get("countries")
    if not isinstance(countries, list) or not countries:
        problems.append("countries must be a non-empty list")
        countries = []
    seen: set = set()
    for i, c in enumerate(countries):
        cid = c.get("code") or f"country {i}"
        if c.get("code") in seen:
            problems.append(f"countries '{cid}': duplicate code")
        seen.add(c.get("code"))
        if not c.get("name"):
            problems.append(f"countries '{cid}': missing name")
        bbox = c.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)):
            problems.append(f"countries '{cid}': numeric bbox[4] required")
        _band(problems, f"countries '{cid}'.replacement_cost_per_m2",
              c.get("replacement_cost_per_m2"))
    return problems


def _point_in_bbox(lat: float, lon: float, bbox: List[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def match_country(lat: float, lon: float,
                  cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The benchmark country whose bbox contains the point.

    Country bounding boxes overlap (a small country can sit inside a
    neighbour's bbox), so the SMALLEST containing bbox wins — the most
    specific match. Returns the country record or None.
    """
    cfg = cfg if cfg is not None else load_benchmarks()
    best: Optional[tuple] = None
    for country in cfg.get("countries") or []:
        bbox = country.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if not _point_in_bbox(lat, lon, bbox):
            continue
        span = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if best is None or span < best[0]:
            best = (span, country)
    return best[1] if best else None


def estimate_exposed_value(buildings_count: float,
                           cost_band: Dict[str, Any],
                           area_band: Dict[str, Any]) -> Dict[str, Any]:
    """Pure: mapped buildings × floor-area-per-building × cost-per-m2.

    Computes the three bands independently (low×low×low, central³,
    high×high×high) so the range honestly compounds the declared
    uncertainties instead of hiding them.
    """
    def band(key: str) -> float:
        return float(buildings_count) * float(area_band[key]) * float(cost_band[key])

    return {
        "low": round(band("low")),
        "central": round(band("central")),
        "high": round(band("high")),
        "unit": "EUR (2025 price context, screening range)",
    }


def loss_screening_estimate(
    lat: float,
    lon: float,
    buildings_count: Optional[float],
    *,
    buildings_source: Optional[str] = None,
    radius_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Full ESTIMATED payload for a point.

    ``buildings_count`` is the real mapped-building count from the calling
    engine (analysis payload or exposure engine) — never fetched here, so
    the function stays pure and offline-testable. Missing/zero counts yield
    an honest ``unavailable``.
    """
    lat, lon = float(lat), float(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"status": "unavailable", "reason": "coordinates out of range"}
    if not buildings_count or buildings_count <= 0:
        return {
            "status": "unavailable",
            "reason": "no mapped building count available for this location "
                      "(OSM completeness varies by region)",
            "claim_status": "ESTIMATED",
        }

    cfg = load_benchmarks()
    country = match_country(lat, lon, cfg)
    defaults = cfg.get("defaults") or {}
    cost_band = (country or {}).get("replacement_cost_per_m2") or \
        defaults.get("replacement_cost_per_m2")
    area_band = defaults.get("floor_area_per_building_m2")

    value = estimate_exposed_value(buildings_count, cost_band, area_band)

    return {
        "status": "ok",
        "claim_status": "ESTIMATED",
        "confidence": Confidence.LOW.value,
        "estimate": {
            "kind": "exposed_value_screening",
            "exposed_value_eur": value,
        },
        "expected_loss": {
            "status": "not_available",
            "statement": EXPECTED_LOSS_STATEMENT,
        },
        "inputs": {
            "buildings_count": {
                "value": float(buildings_count),
                "source": buildings_source or "mapped building count supplied by the calling engine",
                "radius_m": radius_m,
                "caveat": "OpenStreetMap completeness varies by region; counts are a lower bound.",
            },
            "country_benchmark": (
                {"code": country.get("code"), "name": country.get("name")}
                if country else
                {"code": None, "name": "fallback defaults (no country benchmark matched)"}
            ),
            "benchmarks": {
                "replacement_cost_per_m2": cost_band,
                "floor_area_per_building_m2": area_band,
                "config": "config/loss_estimate_benchmarks.json",
            },
        },
        "method": (
            "exposed_value = mapped_buildings × floor_area_per_building × "
            "replacement_cost_per_m2, computed independently for the low, "
            "central and high declared benchmark bands. Benchmarks are declared "
            "screening ranges with stated bases — not valuations."
        ),
        "limitations": [
            "Exposed-value screening, NOT an expected loss: no damage ratio, "
            "no vulnerability model and no hazard footprint is applied.",
            "OSM building counts are mapped features; completeness varies by "
            "region and non-residential buildings are not separated.",
            "Benchmarks are declared screening ranges; country-level licensed "
            "valuation data would narrow them.",
            "The wide low–high span is the honest compound of the declared "
            "input uncertainties.",
        ],
        "separation_note": SEPARATION_NOTE,
        "generated_at": utcnow_iso(),
    }


# ---------------------------------------------------------------------------
# Enriched estimate — pure core + official calibration layers
# ---------------------------------------------------------------------------

_DAMAGE_CURVES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "jrc_damage_curves.json"
)


def load_damage_curves(path: str | None = None) -> Optional[Dict[str, Any]]:
    """Operator-staged depth–damage curves (e.g. transcribed licensed JRC
    values). Staged path, like EM-DAT: parsed when present, None when
    absent — the platform ships no invented curve values."""
    cfg_path = path or os.environ.get("HYDRASHIELD_DAMAGE_CURVES") or _DAMAGE_CURVES_PATH
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def expected_loss_from_depth(exposed_value_eur: Dict[str, Any],
                             depth_m: Optional[float],
                             curve_points: Optional[List[List[float]]]) -> Dict[str, Any]:
    """Pure: exposed value × damage ratio interpolated at ``depth_m``.

    ``curve_points`` are [depth_m, damage_ratio] pairs (ratio 0..1) sorted
    by depth; the ratio clamps at the curve ends. Without depth input or a
    staged curve the expected loss stays honestly ``not_available``.
    """
    if depth_m is None:
        return {"status": "not_available",
                "statement": "No water-depth input is available for this "
                             "location — the expected loss is not computed (declared)."}
    if not curve_points:
        return {"status": "not_available",
                "statement": EXPECTED_LOSS_STATEMENT}
    pts = sorted(curve_points, key=lambda p: p[0])

    def ratio_at(d: float) -> float:
        if d <= pts[0][0]:
            return float(pts[0][1])
        if d >= pts[-1][0]:
            return float(pts[-1][1])
        for (d0, r0), (d1, r1) in zip(pts, pts[1:]):
            if d0 <= d <= d1:
                if d1 == d0:
                    return float(r1)
                return float(r0) + (float(r1) - float(r0)) * (d - d0) / (d1 - d0)
        return float(pts[-1][1])

    ratio = ratio_at(float(depth_m))
    return {
        "status": "ok",
        "claim_status": "ESTIMATED",
        "depth_m": float(depth_m),
        "damage_ratio": round(ratio, 4),
        "expected_loss_eur": {
            "low": round(float(exposed_value_eur.get("low", 0)) * ratio),
            "central": round(float(exposed_value_eur.get("central", 0)) * ratio),
            "high": round(float(exposed_value_eur.get("high", 0)) * ratio),
            "unit": "EUR (screening range)",
        },
        "method": "expected_loss = exposed_value × damage_ratio(depth); ratio "
                  "linearly interpolated on the staged damage curve; the depth "
                  "is caller-supplied, not modelled by Talaix.",
        "limitations": [
            "Screening estimate: a single depth point against an exposed-value "
            "range — not a probabilistic loss, not AAL.",
            "The depth input is external; no flood-depth model is integrated.",
        ],
    }


def enriched_estimate(
    lat: float,
    lon: float,
    buildings_count: Optional[float],
    *,
    buildings_source: Optional[str] = None,
    radius_m: Optional[float] = None,
    depth_m: Optional[float] = None,
) -> Dict[str, Any]:
    """The estimate with its official calibration layers applied:

    1. pure screening estimate (declared benchmarks);
    2. real cadastral floor area where an official cadastre is integrated
       (declared band shape scaled to the real observed mean — basis stated);
    3. Eurostat construction-cost price calibration (official index ratio —
       all bands scaled equally, index values and years printed);
    4. expected loss when a staged damage curve AND a depth input exist —
       otherwise the honest not_available slot.

    The pure ``loss_screening_estimate`` stays the offline-testable core;
    every network-backed layer degrades to an honest declared fallback.
    """
    est = loss_screening_estimate(
        lat, lon, buildings_count,
        buildings_source=buildings_source, radius_m=radius_m)
    if est.get("status") != "ok":
        return est

    inputs = est.setdefault("inputs", {})
    benchmarks = inputs.get("benchmarks") or {}
    cost_band = benchmarks.get("replacement_cost_per_m2") or {}
    declared_area = benchmarks.get("floor_area_per_building_m2") or {}

    # -- 2. real cadastral floor area --------------------------------------
    area_info = None
    try:
        from .cadastre import real_floor_area_m2

        area_info = real_floor_area_m2(lat, lon, radius_m)
    except Exception:
        area_info = None
    if area_info and area_info.get("mean_area_m2"):
        declared_central = float(declared_area.get("central") or 0) or 1.0
        shape_low = float(declared_area.get("low", 0)) / declared_central
        shape_high = float(declared_area.get("high", 0)) / declared_central
        real_mean = float(area_info["mean_area_m2"])
        real_band = {
            "low": round(real_mean * shape_low, 1),
            "central": real_mean,
            "high": round(real_mean * shape_high, 1),
            "basis": ("Real cadastral mean scaled to the declared band shape. "
                      + area_info.get("method", "")),
        }
        est["estimate"]["exposed_value_eur"] = estimate_exposed_value(
            buildings_count, cost_band, real_band)
        inputs["benchmarks"]["floor_area_per_building_m2"] = real_band
        inputs["area_basis"] = {"status": "real_cadastral", **area_info}
    else:
        inputs["area_basis"] = {
            "status": "declared_assumption",
            "note": "No integrated cadastre covers this location — the "
                    "declared floor-area assumption is used (stated in the basis).",
        }

    # -- 3. Eurostat price calibration --------------------------------------
    cfg = load_benchmarks()
    basis_year = int(cfg.get("price_basis_year") or 2023)
    cal: Dict[str, Any]
    try:
        from .eurostat_cci import calibration as cci_calibration

        cal = cci_calibration(
            (inputs.get("country_benchmark") or {}).get("code"),
            basis_year=basis_year)
    except Exception as exc:  # honest degradation — declared bands stand
        cal = {"status": "unavailable", "reason": f"calibration failed: {exc}"}
    if cal.get("status") == "ok":
        factor = float(cal["factor"])
        value = est["estimate"]["exposed_value_eur"]
        for key in ("low", "central", "high"):
            value[key] = round(float(value[key]) * factor)
        value["unit"] = (value.get("unit", "") +
                         f" — price-calibrated to Eurostat {cal['latest_year']} index").strip()
    inputs["price_calibration"] = cal

    # -- 4. expected loss (staged curve + depth input only) -----------------
    curves = load_damage_curves()
    curve_points = None
    curve_meta = None
    if curves:
        curve = (curves.get("curves") or {}).get("flood_residential") or {}
        curve_points = curve.get("points")
        curve_meta = {k: curve.get(k) for k in ("source", "licence_note") if curve.get(k)}
    est["expected_loss"] = expected_loss_from_depth(
        est["estimate"]["exposed_value_eur"], depth_m, curve_points)
    if curve_meta and est["expected_loss"].get("status") == "ok":
        est["expected_loss"]["curve"] = curve_meta
    return est
