"""
Talaix Supply Chain Origin & EUDR Evidence engine.

No Flask imports. Screens origin/green claims using the real datasets that
exist in this repository:

* ESA WorldCover 10 m 2021 snapshot — ``src.gis_mapping.landcover.fetch_landcover``
* Sentinel-2 NDVI/NDMI — ``src.dashboard.real_data.fetch_satellite_data``
* Hansen/UMD Global Forest Change 2023 v1.11 — ``src.gis_mapping.forest_loss.fetch_forest_loss``

The engine screens for deforestation evidence through 2023 but never certifies
a claim as "green", "deforestation-free", or EUDR-compliant.

Verdict vocabulary:

* Per plot: ``partial_evidence`` / ``no_evidence``
* Per claim:
  * ``screened_findings_detected`` — post-cutoff tree-cover loss detected
  * ``no_inconsistency_detected_with_current_evidence`` — all plots assessed,
    no post-cutoff loss detected
  * ``not_verifiable_with_current_evidence`` — assessment incomplete
* Deforestation assessment: ``no_loss_detected``,
  ``no_post_cutoff_loss_detected``, ``loss_detected_after_cutoff``, or
  ``not_verifiable`` when the layer is unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..dashboard.real_data import fetch_satellite_data
from ..gis_mapping.forest_loss import fetch_forest_loss
from ..gis_mapping.landcover import fetch_landcover
from .evidence import EvidenceRecord, content_hash, utcnow_iso
from .tx_seal import issue_seal

ENGINE_VERSION = "1.0.0"
EUDR_CUTOFF_DATE = "2020-12-31"

#: Commodities covered by Regulation (EU) 2023/1115 (plus derived products).
EUDR_COMMODITIES = ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]

#: Common caller spellings mapped onto the EUDR vocabulary.
_COMMODITY_ALIASES = {"soy": "soya", "palm oil": "oil_palm", "palm_oil": "oil_palm"}

SUPPLIER_DECLARATION = (
    "Supplier, commodity and country metadata is declared by the caller — "
    "not verified by Talaix."
)

SUPPLY_CHAIN_FRAMEWORKS = [
    {
        "id": "eudr",
        "name": "EU Deforestation Regulation (EUDR)",
        "aspect": "Due diligence on deforestation and forest degradation",
        "role": "regulatory context",
        "note": (
            "EUDR requires proof that products are deforestation-free after "
            "31 December 2020. Talaix screens with Hansen/UMD GFC through 2023 "
            "and reports what the data show; it does not verify compliance."
        ),
    },
    {
        "id": "uk_frcs",
        "name": "UK Forest Risk Commodity Scheme",
        "aspect": "Due diligence on forest-risk commodities in UK supply chains",
        "role": "regulatory context",
        "note": (
            "Operator-level due-diligence evidence can be supported, including "
            "Hansen/UMD GFC forest-loss screening through 2023, but this is not "
            "a compliance verification."
        ),
    },
    {
        "id": "green_claims",
        "name": "Green Claims / substantiation",
        "aspect": "Advertising and product-level green claims",
        "role": "commercial context",
        "note": (
            "The engine reports the evidence that exists and the evidence that "
            "is missing. Hansen/UMD GFC provides a 2001–2023 forest-loss time "
            "series, but a green or deforestation-free label still requires "
            "audit-grade evidence beyond remote sensing."
        ),
    },
]

WORLD_COVER_LIMITATION = (
    "ESA WorldCover 10m 2021 v200 is a single-year snapshot. It shows the "
    "dominant land-cover class at one point in time, but it cannot detect "
    "forest loss, conversion, or compliance with the EUDR cutoff date."
)

SENTINEL_LIMITATION = (
    "Sentinel-2 NDVI/NDMI is a recent cloud-free optical observation when "
    "available. It cannot prove deforestation-free status and may be "
    "unavailable due to cloud cover, revisit gaps, or missing Copernicus "
    "credentials."
)

FOREST_LOSS_PRODUCT = "Hansen/UMD Global Forest Change 2023 v1.11 (GFC)"
FOREST_LOSS_LIMITATION = (
    "Hansen/UMD GFC is a 30 m spatial-resolution layer. It misses small "
    "clearings and degradation, and it counts a pixel as forested only when "
    "the 2000 canopy cover is at least 30%."
)
FOREST_LOSS_VINTAGE_LIMITATION = (
    "Hansen/UMD GFC 2023 v1.11 covers tree-cover loss through 2023 only. "
    "Loss in 2024 or later is not included and must be declared as a vintage limitation."
)

DISCLAIMER = (
    "Talaix Supply Chain Origin Evidence is a screening-level data product. "
    "It is NOT a EUDR compliance verification, NOT a deforestation-free "
    "certificate, and NOT a supply-chain audit. Verdicts are evidence "
    "screening only; any green or deforestation-free claim remains "
    "unverified with the current datasets."
)

HONESTY_CONTRACT = (
    "Unavailable data is declared, never invented. The engine screens with "
    "Hansen/UMD GFC through 2023, states that land cover is a single-year "
    "snapshot, reports when Sentinel-2 or GFC is unavailable, and never "
    "claims EUDR compliance or deforestation-free status."
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f
    except (TypeError, ValueError):
        return None


def _normalise_plot(plot: Any, index: int) -> Dict[str, Any]:
    """Return a validated plot dict; lat/lon may be None if geocoding is needed."""
    if not isinstance(plot, dict):
        return {
            "name": f"plot_{index + 1}",
            "lat": None,
            "lon": None,
            "address": None,
            "error": "plot must be an object",
        }
    lat = _safe_float(plot.get("lat"))
    lon = _safe_float(plot.get("lon"))
    address = (plot.get("address") or "").strip() or None
    return {
        "name": (plot.get("name") or "").strip() or f"plot_{index + 1}",
        "lat": lat,
        "lon": lon,
        "address": address,
        "error": None,
    }


def _geocode_plot(plot: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an address-only plot; keeps honest errors in the plot record."""
    from ..dashboard.real_data import geocode_location

    address = plot.get("address")
    if not address:
        plot["error"] = "plot has no lat/lon and no address"
        return plot
    geo = geocode_location(address)
    if "error" in geo:
        plot["error"] = f"geocoding failed: {geo['error']}"
        return plot
    plot["lat"] = geo.get("lat")
    plot["lon"] = geo.get("lon")
    return plot


def _build_deforestation_assessment(forest_loss: Dict[str, Any]) -> Dict[str, Any]:
    """Map a GFC result to a structured deforestation assessment."""
    if "error" in forest_loss:
        return {
            "status": "not_verifiable",
            "reason": f"Forest-loss layer unavailable: {forest_loss['error']}",
            "cutoff_date": EUDR_CUTOFF_DATE,
        }

    loss_years = forest_loss.get("loss_years") or {}
    latest = forest_loss.get("latest_loss_year")
    fraction = forest_loss.get("loss_pixel_fraction", 0.0)
    post_fraction = forest_loss.get("loss_after_2020_pixel_fraction", 0.0)
    vintage = forest_loss.get("vintage_note", FOREST_LOSS_VINTAGE_LIMITATION)

    if forest_loss.get("loss_after_2020"):
        years = ", ".join(str(y) for y in sorted(loss_years.keys()) if y >= 2021)
        return {
            "status": "loss_detected_after_cutoff",
            "finding": (
                f"Tree-cover loss detected in {years} after the EUDR cutoff "
                f"2020-12-31 ({post_fraction:.1%} of the screened window). "
                "Screening evidence — not a legality determination."
            ),
            "latest_loss_year": latest,
            "loss_years": loss_years,
            "loss_pixel_fraction": fraction,
            "loss_after_2020_pixel_fraction": post_fraction,
            "cutoff_date": EUDR_CUTOFF_DATE,
            "vintage_note": vintage,
        }

    if forest_loss.get("loss_detected"):
        return {
            "status": "no_post_cutoff_loss_detected",
            "finding": (
                f"Tree-cover loss detected only in earlier years "
                f"({latest}); no loss after the EUDR cutoff through 2023."
            ),
            "latest_loss_year": latest,
            "loss_years": loss_years,
            "loss_pixel_fraction": fraction,
            "loss_after_2020_pixel_fraction": post_fraction,
            "cutoff_date": EUDR_CUTOFF_DATE,
            "vintage_note": vintage,
        }

    return {
        "status": "no_loss_detected",
        "finding": "No tree-cover loss detected in the screened window through 2023.",
        "latest_loss_year": None,
        "loss_years": {},
        "loss_pixel_fraction": 0.0,
        "loss_after_2020_pixel_fraction": 0.0,
        "cutoff_date": EUDR_CUTOFF_DATE,
        "vintage_note": vintage,
    }


def _assess_plot(lat: float, lon: float, name: Optional[str] = None) -> Dict[str, Any]:
    """Screen one plot with land cover, Sentinel-2 and Hansen/UMD GFC."""
    evidence: List[Dict[str, Any]] = []
    limitations: List[str] = []
    partial = False

    landcover = fetch_landcover(lat, lon)
    if "error" not in landcover:
        partial = True
        evidence.append(
            EvidenceRecord.open_data(
                source=landcover.get("source", "ESA WorldCover"),
                status="OBSERVED",
                temporal="OBSERVED",
                dataset="ESA WorldCover 10m 2021 v200",
                resolution=landcover.get("resolution"),
                method="dominant land-cover class in ~1 km window",
                limitations=WORLD_COVER_LIMITATION,
                location={"lat": lat, "lon": lon},
            ).to_dict()
        )
        limitations.append(WORLD_COVER_LIMITATION)
    else:
        limitations.append(f"Land-cover lookup unavailable: {landcover.get('error')}")

    satellite = fetch_satellite_data(lat, lon, days_back=60)
    if "error" not in satellite:
        partial = True
        resolution = satellite.get("resolution_m")
        evidence.append(
            EvidenceRecord.satellite(
                source=satellite.get("source", "Sentinel-2 L2A"),
                status="OBSERVED",
                temporal="OBSERVED",
                dataset="Sentinel-2 L2A",
                resolution=f"{resolution} m" if resolution else "10 m",
                method="NDVI/NDMI from cloud-free scene",
                acquired_at=satellite.get("observation_date"),
                limitations=SENTINEL_LIMITATION,
                location={"lat": lat, "lon": lon},
            ).to_dict()
        )
        obs_date = (satellite.get("observation_date") or "")[:10]
        if obs_date and obs_date >= EUDR_CUTOFF_DATE:
            limitations.append(
                f"Sentinel-2 observation ({obs_date}) is after the EUDR cutoff; "
                f"it cannot demonstrate the area was not deforested before {EUDR_CUTOFF_DATE}."
            )
        elif obs_date:
            limitations.append(
                f"Sentinel-2 observation ({obs_date}) predates the EUDR cutoff "
                "and cannot assess current status."
            )
    else:
        limitations.append(f"Sentinel-2 observation unavailable: {satellite.get('error')}")

    forest_loss = fetch_forest_loss(lat, lon)
    deforestation_assessment = _build_deforestation_assessment(forest_loss)
    if "error" not in forest_loss:
        partial = True
        evidence.append(
            EvidenceRecord.open_data(
                source=forest_loss.get("source", FOREST_LOSS_PRODUCT),
                status="OBSERVED",
                temporal="HISTORICAL",
                dataset=forest_loss.get("dataset", FOREST_LOSS_PRODUCT),
                resolution=forest_loss.get("resolution"),
                method="Hansen/UMD GFC windowed screening (~1 km box)",
                limitations=f"{FOREST_LOSS_LIMITATION} {forest_loss.get('vintage_note', FOREST_LOSS_VINTAGE_LIMITATION)}",
                location={"lat": lat, "lon": lon},
            ).to_dict()
        )
        limitations.append(forest_loss.get("vintage_note", FOREST_LOSS_VINTAGE_LIMITATION))
        limitations.append(FOREST_LOSS_LIMITATION)
    else:
        limitations.append(f"Forest-loss layer unavailable: {forest_loss.get('error')}")

    return {
        "name": name,
        "lat": lat,
        "lon": lon,
        "verdict": "partial_evidence" if partial else "no_evidence",
        "landcover": landcover,
        "satellite": satellite,
        "forest_loss": forest_loss,
        "deforestation_assessment": deforestation_assessment,
        "evidence": evidence,
        "limitations": limitations,
    }


def evaluate_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate an origin/green claim for supply-chain screening.

    Input schema (all optional except at least one plot attempt):
        {
          "supplier": "Acme S.A.",
          "commodity": "soy",
          "country": "Brazil",
          "plots": [
            {"name": "Farm A", "lat": -12.3, "lon": -55.4},
            {"name": "Farm B", "address": "Some place"}
          ]
        }

    Returns a claim result with the strict honesty contract described in the
    module docstring.
    """
    supplier = str(claim.get("supplier") or "").strip() or None
    commodity = str(claim.get("commodity") or "").strip() or None
    country = str(claim.get("country") or "").strip() or None
    raw_plots = claim.get("plots")
    if not isinstance(raw_plots, list):
        raw_plots = []

    # Map the caller's commodity spelling onto the EUDR vocabulary and flag
    # anything outside the regulated list — advisory screen only, never a block.
    commodity_normalised: Optional[str] = None
    commodity_advisory: Optional[str] = None
    if commodity:
        commodity_normalised = _COMMODITY_ALIASES.get(commodity.lower(), commodity.lower())
        if commodity_normalised not in EUDR_COMMODITIES:
            commodity_advisory = (
                f"'{commodity}' is not an EUDR-covered commodity "
                f"({', '.join(EUDR_COMMODITIES)}); advisory screen only."
            )

    normalised_plots: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_plots):
        plot = _normalise_plot(raw, idx)
        if plot["lat"] is None or plot["lon"] is None:
            if plot["error"] is None:
                plot = _geocode_plot(plot)
        normalised_plots.append(plot)

    plot_results: List[Dict[str, Any]] = []
    for plot in normalised_plots:
        if plot.get("error"):
            plot_results.append({
                "name": plot.get("name"),
                "lat": plot.get("lat"),
                "lon": plot.get("lon"),
                "verdict": "no_evidence",
                "landcover": {"error": plot["error"]},
                "satellite": {"error": plot["error"]},
                "evidence": [],
                "limitations": [plot["error"]],
            })
            continue
        try:
            plot_results.append(
                _assess_plot(plot["lat"], plot["lon"], name=plot.get("name"))
            )
        except Exception as exc:  # noqa: BLE001 — honesty path below
            plot_results.append({
                "name": plot.get("name"),
                "lat": plot.get("lat"),
                "lon": plot.get("lon"),
                "verdict": "no_evidence",
                "landcover": {"error": str(exc)},
                "satellite": {"error": str(exc)},
                "evidence": [],
                "limitations": [f"Plot assessment failed: {exc}"],
            })

    partial_count = sum(1 for p in plot_results if p["verdict"] == "partial_evidence")
    no_evidence_count = len(plot_results) - partial_count

    # Aggregate per-plot deforestation assessments into a claim-level view.
    statuses = [p["deforestation_assessment"]["status"] for p in plot_results]
    any_post_cutoff = any(s == "loss_detected_after_cutoff" for s in statuses)
    all_assessed = statuses and all(s != "not_verifiable" for s in statuses)

    if any_post_cutoff:
        claim_verdict = "screened_findings_detected"
    elif all_assessed:
        claim_verdict = "no_inconsistency_detected_with_current_evidence"
    else:
        claim_verdict = "not_verifiable_with_current_evidence"

    if any_post_cutoff:
        top_deforestation = {
            "status": "loss_detected_after_cutoff",
            "reason": (
                "At least one screened plot shows Hansen/UMD GFC tree-cover loss "
                f"after the EUDR cutoff {EUDR_CUTOFF_DATE}. This is screening evidence, "
                "not a legality determination."
            ),
            "cutoff_date": EUDR_CUTOFF_DATE,
        }
    elif any(s == "no_post_cutoff_loss_detected" for s in statuses):
        top_deforestation = {
            "status": "no_post_cutoff_loss_detected",
            "reason": (
                "Tree-cover loss was detected only before the EUDR cutoff; "
                "no post-cutoff loss through 2023."
            ),
            "cutoff_date": EUDR_CUTOFF_DATE,
        }
    elif any(s == "no_loss_detected" for s in statuses):
        top_deforestation = {
            "status": "no_loss_detected",
            "reason": "No Hansen/UMD GFC tree-cover loss detected in assessed plots through 2023.",
            "cutoff_date": EUDR_CUTOFF_DATE,
        }
    else:
        top_deforestation = {
            "status": "not_verifiable",
            "reason": "Forest-loss layer unavailable for all screened plots.",
            "cutoff_date": EUDR_CUTOFF_DATE,
        }

    declared_gaps: List[Dict[str, Any]] = [
        {
            "type": "vintage_limitation",
            "dataset": FOREST_LOSS_PRODUCT,
            "reason": FOREST_LOSS_VINTAGE_LIMITATION,
        },
        {
            "type": "single_year_snapshot",
            "dataset": "ESA WorldCover 10m 2021 v200",
            "reason": WORLD_COVER_LIMITATION,
        },
    ]

    if any("Forest-loss layer unavailable" in lim for p in plot_results for lim in p["limitations"]):
        declared_gaps.append({
            "type": "data_unavailable",
            "dataset": FOREST_LOSS_PRODUCT,
            "reason": "Hansen/UMD GFC fetch failed for at least one plot.",
        })

    if any("Sentinel-2 observation unavailable" in lim for p in plot_results for lim in p["limitations"]):
        declared_gaps.append({
            "type": "data_unavailable",
            "dataset": "Sentinel-2 L2A",
            "reason": SENTINEL_LIMITATION,
        })

    if not plot_results:
        declared_gaps.append({
            "type": "no_plots",
            "dataset": None,
            "reason": "No valid production plots were supplied.",
        })

    claim_basis = {
        "supplier": supplier,
        "commodity": commodity,
        "country": country,
        "plots": [
            {"name": p.get("name"), "lat": p.get("lat"), "lon": p.get("lon")}
            for p in plot_results
        ],
    }
    claim_id = content_hash(claim_basis)[:16]

    certification_statement = (
        "This report is not a EUDR due-diligence statement, not a "
        "deforestation-free certificate, and not a supply-chain audit. It is a "
        "screening-level evidence summary only."
    )

    screening_note = (
        f"Screened {len(plot_results)} plot(s) with ESA WorldCover, Sentinel-2, "
        f"and {FOREST_LOSS_PRODUCT}. Post-cutoff loss status is based on GFC "
        f"lossyear codes {21}–{23} (2021–2023)."
    )

    return {
        "claim_id": claim_id,
        "supplier": supplier,
        "commodity": commodity,
        "commodity_normalised": commodity_normalised,
        "commodity_advisory": commodity_advisory,
        "country": country,
        "supplier_declaration": SUPPLIER_DECLARATION,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "frameworks": SUPPLY_CHAIN_FRAMEWORKS,
        "eudr_cutoff_date": EUDR_CUTOFF_DATE,
        "claim_verdict": claim_verdict,
        "deforestation_assessment": top_deforestation,
        "eudr_timeline_note": (
            "EUDR requires that production land was not subject to deforestation "
            f"or forest degradation after {EUDR_CUTOFF_DATE}. Hansen/UMD GFC 2023 "
            "v1.11 supports screening through 2023; 2024+ loss is not covered."
        ),
        "screening_note": screening_note,
        "certification_statement": certification_statement,
        "plot_count": len(plot_results),
        "partial_evidence_count": partial_count,
        "no_evidence_count": no_evidence_count,
        "plots": plot_results,
        "declared_gaps": declared_gaps,
        "honesty_contract": HONESTY_CONTRACT,
        "disclaimer": DISCLAIMER,
        "authenticity": issue_seal("supplychain", claim_id, claim_basis),
    }
