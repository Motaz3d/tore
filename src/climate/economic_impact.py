"""
Economic Impact Engine v1 (docs/ECONOMIC_INTELLIGENCE.md — no-fake-money rule).

Formalises the three strictly separated economic blocks for a location:

- ``observed_losses`` — ``unavailable`` when no documented loss dataset covers
  the queried context; ``ok`` when an integrated free source (NOAA NCEI
  Billion-Dollar Disasters) has real documented US national figures. Every
  figure is tagged with source, reference period, geographic scope, licence
  note and claim status. Never populated with invented figures.
- ``modelled_estimates`` — the exposure-bounded qualitative profile from
  ``exposure_econ.build_economic_exposure`` (real mapped categories, counts
  and caveats) with the mandatory ``monetary_quantification:
  not_quantified`` statement (exact wording, doc §3).
- ``projections`` — always ``not_available``: economic projections require
  scenario-labelled datasets that are not integrated.

Absolute norm: Talaix does not output euro/dollar loss figures unless a
documented valuation dataset with a stated method is integrated. Everything
economic here carries ``confidence: low``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..dashboard.cache import cached
from .evidence import EvidenceRecord, utcnow_iso
from .exposure_econ import NOT_QUANTIFIED_STATEMENT, build_economic_exposure
from .losses import OBSERVED_LOSSES_STATEMENT, documented_loss_figures
from .ontology import ClaimStatus, Confidence

TTL_ECON_IMPACT = 3600.0  # 1 h

OBSERVED_LOSSES_RESEARCH_CANDIDATES = [
    "EM-DAT international disaster database (documented event losses) — staged ingest; operator export required",
    "DesInventar national disaster loss databases — staged ingest; operator export required",
    "Insurance/reinsurance loss databases (e.g. Munich Re NatCatSERVICE, Swiss Re sigma) — planned after first revenue",
]

PROJECTIONS_STATEMENT = (
    "Economic projections require scenario-labelled datasets not yet integrated."
)


@cached("economic_impact", TTL_ECON_IMPACT)
def assess_economic_impact(lat: float, lon: float) -> Dict[str, Any]:
    """Economic-impact formalization for a point. Cached 1 h.

    Three strictly separated blocks — observed losses, modelled estimates,
    projections — plus evidence. Observed losses now include real documented
    figures when the queried location is within the geographic coverage of an
    integrated source (NOAA NCEI, US only). No monetary values are invented.
    """

    lat, lon = float(lat), float(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Coordinates out of range"}

    evidence: List[Dict[str, Any]] = []

    # -- observed losses (documented figures when available; never fabricated)
    loss_data = documented_loss_figures(for_lat=lat, for_lon=lon)
    if loss_data["status"] == "ok" and loss_data.get("figures"):
        observed_losses = {
            "status": "ok",
            "statement": f"{loss_data['figure_count']} documented loss figure(s) "
                         "available for the queried context.",
            "figures": loss_data["figures"],
            "sources": loss_data["sources"],
            "monetary_quantification": {
                "status": "documented",
                "statement": (
                    "Values are documented losses from integrated free sources "
                    "with stated reference periods and licence notes; they are "
                    "national aggregates, not point-specific loss estimates."
                ),
            },
            "confidence": Confidence.LOW.value,
        }
        evidence.append(EvidenceRecord.open_data(
            "NOAA NCEI Billion-Dollar Weather and Climate Disasters",
            status=ClaimStatus.DOCUMENTED.value,
            temporal="HISTORICAL",
            reference_period={"start": "1980", "end": "2021"},
            provider_url="https://www.ncei.noaa.gov/access/billions/",
            license="US government public data; cite NOAA NCEI as source.",
            limitations=(
                "US national aggregates from state-level source data; "
                "multi-state events may be summed across affected states. "
                "No point-specific valuation is implied."
            ),
            location={"lat": lat, "lon": lon},
        ).to_dict())
    else:
        observed_losses = {
            "status": "unavailable",
            "statement": OBSERVED_LOSSES_STATEMENT,
            "reason": loss_data.get("reason"),
            "research_candidates": OBSERVED_LOSSES_RESEARCH_CANDIDATES,
            "confidence": Confidence.LOW.value,
        }
        evidence.append(EvidenceRecord.unknown(
            "Talaix economic impact engine",
            why=(loss_data.get("reason") or OBSERVED_LOSSES_STATEMENT)
                + " Documented loss datasets (EM-DAT, DesInventar, insurance loss "
                  "databases) are staged or planned, not integrated for this context.",
            location={"lat": lat, "lon": lon},
        ).to_dict())

    # -- modelled estimates: exposure-bounded qualitative profile -------------
    exposure = build_economic_exposure(lat, lon)
    exposure_ok = "error" not in exposure
    if exposure_ok:
        categories = exposure.get("exposure") or {}
        modelled_estimates = {
            "status": "ok",
            "basis": "Exposure-bounded qualitative profile: real mapped "
                     "exposure categories and counts bound what could be "
                     "affected; no susceptibility or valuation model exists.",
            "exposure_profile": categories,
            "analysis_window": exposure.get("analysis_window"),
            "radius_km": exposure.get("radius_km"),
            "monetary_quantification": {
                "status": "not_quantified",
                "statement": NOT_QUANTIFIED_STATEMENT,
            },
            "caveats": sorted({
                str(v.get("completeness_caveat"))
                for v in categories.values()
                if isinstance(v, dict) and v.get("completeness_caveat")
            }),
            "confidence": Confidence.LOW.value,
        }
        evidence.extend((exposure.get("provenance") or {}).get("evidence") or [])
    else:
        modelled_estimates = {
            "status": "unavailable",
            "reason": exposure.get("error"),
            "monetary_quantification": {
                "status": "not_quantified",
                "statement": NOT_QUANTIFIED_STATEMENT,
            },
            "confidence": Confidence.LOW.value,
        }
        evidence.append(EvidenceRecord.unknown(
            "OpenStreetMap (ohsome / Overpass) / ESA WorldCover",
            why=str(exposure.get("error")),
            location={"lat": lat, "lon": lon},
        ).to_dict())

    # -- Talaix loss screening estimate (ESTIMATED — strictly separated) -----
    # The engine's own monetary function: computed from the SAME real mapped
    # building count above and declared benchmarks. Never merged with the
    # documented observed losses or the qualitative exposure profile.
    from .loss_estimate import enriched_estimate

    buildings = buildings_source = None
    radius_m = None
    if exposure_ok:
        buildings = (categories.get("buildings") or {}).get("count")
        buildings_source = (categories.get("buildings") or {}).get("source")
        radius_m = (exposure.get("radius_km") or 0) * 1000 or None
    loss_screening = enriched_estimate(
        lat, lon, buildings,
        buildings_source=buildings_source, radius_m=radius_m)

    # -- projections (structurally separated, never invented) -----------------
    projections = {
        "status": "not_available",
        "statement": PROJECTIONS_STATEMENT,
        "confidence": Confidence.LOW.value,
    }

    return {
        "status": "ok" if exposure_ok else "partial",
        "location": {"lat": lat, "lon": lon},
        "generated_at": utcnow_iso(),
        "observed_losses": observed_losses,
        "modelled_estimates": modelled_estimates,
        "projections": projections,
        "loss_screening_estimate": loss_screening,
        "evidence": evidence,
        "confidence": Confidence.LOW.value,
        "separation_note": (
            "The blocks are strictly separated: observed losses "
            "(documented figures from integrated sources when available), "
            "modelled estimates (exposure-bounded qualitative profile — no "
            "monetary values), the Talaix loss screening estimate (ESTIMATED "
            "exposed-value range computed from the real mapped building "
            "count and declared benchmarks) and projections "
            "(scenario-conditioned — none integrated). They are never merged "
            "into a single figure."
        ),
        "limitations": [
            "No monetary loss, premium or market-size figure is produced "
            "anywhere in this payload unless explicitly sourced and tagged.",
            "Observed losses are geographic-coverage bounded; US national "
            "aggregates are not point-specific loss estimates.",
            "The loss screening estimate is an ESTIMATED exposed-value "
            "screening range from declared benchmarks — not an expected "
            "loss, not a valuation.",
            "Exposure counts are mapped OpenStreetMap features; completeness "
            "varies by region.",
        ],
        "provenance": {
            "engine": "economic_impact_v1",
            "research": [
                {"id": "ipccar6wg2",
                 "role": "risk framework: hazard x exposure x vulnerability"},
            ],
            "exposure_source": "src/climate/exposure_econ.py (OSM/ohsome + "
                               "Overpass counts, ESA WorldCover)",
            "loss_source": "src/climate/losses.py (NOAA NCEI billion-dollar "
                           "disasters; staged EM-DAT / DesInventar)",
            "no_fake_money_rule": "docs/ECONOMIC_INTELLIGENCE.md §3",
        },
    }
