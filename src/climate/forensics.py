"""
Talaix Environmental Security & Forensic Verification engine.

No Flask imports. Builds a content-hashed "Environmental Forensic Evidence Pack"
for investigators: given a site and a structured claim, it cross-matches the
claim against observed physical evidence (satellite, land cover, active fires,
Hansen/UMD GFC forest-loss time series) and documents consistency /
inconsistency / cannot_assess.

The engine NEVER determines legality, illegality, guilt or criminal conduct.
It is an evidence annex for qualified investigators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..dashboard.real_data import fetch_active_fires, fetch_satellite_data, geocode_location
from ..gis_mapping.forest_loss import fetch_forest_loss
from ..gis_mapping.landcover import fetch_landcover
from .evidence import EvidenceRecord, content_hash, utcnow_iso
from .tx_seal import issue_seal

ENGINE_VERSION = "1.0.0"

CASE_TYPOLOGIES: List[Dict[str, str]] = [
    {
        "id": "illegal_logging",
        "label": "Suspected unauthorised timber extraction",
        "note": (
            "Relevant evidence: land cover (forest presence), Hansen/UMD GFC "
            "forest-loss time series through 2023, active fires "
            "(burning associated with clearing), Sentinel-2 NDVI. Missing: "
            "high-resolution concession boundaries and timber-chain-of-custody "
            "documents."
        ),
    },
    {
        "id": "illegal_mining",
        "label": "Suspected unauthorised extraction / land disturbance",
        "note": (
            "Relevant evidence: land cover change proxy (single snapshot), "
            "Sentinel-2 NDVI/NDWI. Missing: a dedicated mining/disturbance "
            "detection dataset; concession boundaries; ground inspection."
        ),
    },
    {
        "id": "unlicensed_clearing",
        "label": "Land clearing without a permit",
        "note": (
            "Relevant evidence: land cover class, Hansen/UMD GFC forest-loss "
            "time series through 2023, active-fire detections, Sentinel-2 "
            "vegetation signal. Missing: legal permit status of the site."
        ),
    },
    {
        "id": "waste_dumping",
        "label": "Illegal waste disposal",
        "note": (
            "Relevant evidence: land cover and Sentinel-2 spectral indices "
            "(limited). Missing: a dedicated waste/dump detection dataset; "
            "ground inspection; regulatory records."
        ),
    },
    {
        "id": "other",
        "label": "Other environmental-crime suspicion",
        "note": (
            "The investigator supplies a free-text subject claim. Relevant "
            "evidence depends on the claim. Missing layers are declared "
            "explicitly in the pack."
        ),
    },
]

CLAIM_TYPES: List[Dict[str, str]] = [
    {"id": "site_forested", "label": "The site is forested / intact"},
    {"id": "no_recent_clearing", "label": "No recent tree-cover clearing (e.g., post-2020)"},
    {"id": "no_burning", "label": "No open burning occurs at the site"},
    {"id": "vegetation_present", "label": "Vegetation / restoration is present and active"},
    {"id": "free_text", "label": "Any other claim (investigator assessment)"},
]

FORENSIC_FRAMEWORKS: List[Dict[str, str]] = [
    {
        "id": "fatf_environmental_crime",
        "name": "FATF \"Money Laundering from Environmental Crime\" (2021)",
        "aspect": "Financial-crime context for environmental offences",
        "role": "financial-crime context",
        "note": (
            "This pack can serve as the physical-evidence annex to an AML/FIU "
            "environmental-crime case file. Talaix holds NO financial transaction "
            "data; that boundary is declared explicitly."
        ),
    },
    {
        "id": "eu_environmental_crime_directive",
        "name": "EU Environmental Crime Directive (EU) 2024/1203",
        "aspect": "Criminal environmental conduct",
        "role": "legal context",
        "note": (
            "The Directive defines categories of environmental crime. Talaix "
            "does not determine whether conduct is criminal; it documents "
            "observed physical evidence only."
        ),
    },
    {
        "id": "interpol_unep",
        "name": "INTERPOL / UNEP environmental-crime enforcement",
        "aspect": "Operational enforcement support",
        "role": "enforcement context",
        "note": (
            "Investigators can use the pack as a structured, content-hashed "
            "starting point; it does not replace warrants, seizures or expert "
            "forensic analysis."
        ),
    },
]

LEGAL_NOTE = (
    "Talaix does not determine legality, illegality, guilt or criminal conduct. "
    "This pack documents physical evidence and claim–evidence consistency for "
    "qualified investigators. It is not evidence of a crime and not exoneration."
)

FORENSICS_DISCLAIMER = (
    "Talaix Environmental Forensic Evidence Packs are screening-level products. "
    "They are not a forensic-lab certification, not a legal opinion, and not a "
    "substitute for ground inspection or official investigation. Observations are "
    "limited to the declared dataset coverage per layer. Talaix holds no financial "
    "transaction data."
)

HONESTY_CONTRACT = (
    "Every evidence item is typed, sourced and content-hashed. Hansen/UMD GFC "
    "forest-loss time series through 2023 is integrated; remaining gaps "
    "(mining/waste detection, financial data, 2024+ loss) are declared. "
    "Fetch failures are reported as unavailable. The engine never labels "
    "a site or claim as legal, illegal, criminal or exonerated."
)

_FOREST_LOSS_PRODUCT = "Hansen/UMD Global Forest Change 2023 v1.11 (GFC)"
_FOREST_LOSS_VINTAGE_LIMITATION = (
    "Hansen/UMD GFC 2023 v1.11 covers tree-cover loss through 2023 only. "
    "Loss in 2024 or later is not included and must be declared as a vintage limitation."
)

_FINANCIAL_DATA_BOUNDARY = (
    "Talaix does not hold or process financial transaction data. Any money-laundering "
    "or proceeds-of-crime assessment requires separate financial analysis by an "
    "FIU or competent authority."
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _typology_by_id(typology_id: str) -> Optional[Dict[str, str]]:
    for t in CASE_TYPOLOGIES:
        if t["id"] == typology_id:
            return t
    return None


def _claim_type_by_id(claim_type_id: str) -> Optional[Dict[str, str]]:
    for c in CLAIM_TYPES:
        if c["id"] == claim_type_id:
            return c
    return None


def _resolve_site(site: Any) -> Dict[str, Any]:
    """Return a validated/normalised site dict, with an honest error if unresolved."""
    if not isinstance(site, dict):
        return {"error": "site must be an object"}

    lat = _safe_float(site.get("lat"))
    lon = _safe_float(site.get("lon"))
    name = (site.get("name") or "").strip() or None
    address = (site.get("address") or "").strip() or None

    if lat is not None and lon is not None:
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"error": "lat/lon out of range"}
        return {"name": name, "lat": lat, "lon": lon}

    if address:
        geo = geocode_location(address)
        if "error" in geo:
            return {"error": f"geocoding failed: {geo['error']}"}
        return {"name": name or geo.get("name"), "lat": geo["lat"], "lon": geo["lon"]}

    return {"error": "site must include lat/lon or a resolvable address"}


def _normalise_reference_documents(docs: Any) -> List[Dict[str, str]]:
    """Keep reference documents as declared links; never fetch them."""
    normalised: List[Dict[str, str]] = []
    if not isinstance(docs, list):
        return normalised
    for doc in docs:
        if isinstance(doc, dict):
            title = str(doc.get("title") or "").strip()
            url = str(doc.get("url") or "").strip()
            if title or url:
                normalised.append({"title": title or "Untitled document", "url": url})
    return normalised


def _record_with_hash(record: EvidenceRecord) -> Dict[str, Any]:
    """Expose the record's own content hash as a chain-of-custody field."""
    d = record.to_dict()
    d["content_hash"] = d.get("evidence_id")
    return d


def _fetch_evidence_bundle(
    lat: float,
    lon: float,
    radius_km: float = 25.0,
) -> Dict[str, Any]:
    """Gather every fetcher that exists; failures become declared gaps, not exceptions."""
    evidence_records: List[Dict[str, Any]] = []
    bundle: Dict[str, Any] = {"landcover": {}, "satellite": {}, "active_fires": {}, "forest_loss": {}}

    try:
        landcover = fetch_landcover(lat, lon)
        bundle["landcover"] = landcover
        if "error" not in landcover:
            evidence_records.append(_record_with_hash(EvidenceRecord.open_data(
                source=landcover.get("source", "ESA WorldCover"),
                status="OBSERVED",
                temporal="OBSERVED",
                dataset="ESA WorldCover 10m 2021 v200",
                resolution=landcover.get("resolution"),
                method="dominant land-cover class in ~1 km window",
                location={"lat": lat, "lon": lon},
                limitations="Single-year snapshot; cannot detect forest loss or legal status.",
            )))
    except Exception as exc:  # noqa: BLE001
        bundle["landcover"] = {"error": f"Land-cover fetch raised {type(exc).__name__}: {exc}"}

    try:
        satellite = fetch_satellite_data(lat, lon, days_back=60)
        bundle["satellite"] = satellite
        if "error" not in satellite:
            evidence_records.append(_record_with_hash(EvidenceRecord.satellite(
                source=satellite.get("source", "Sentinel-2 L2A"),
                status="OBSERVED",
                temporal="OBSERVED",
                dataset="Sentinel-2 L2A",
                resolution=f"{satellite.get('resolution_m', 10)} m" if satellite.get("resolution_m") else "10 m",
                method="NDVI/NDMI/NDWI from cloud-free scene",
                acquired_at=satellite.get("observation_date"),
                location={"lat": lat, "lon": lon},
                limitations="Optical sensor; unavailable under persistent cloud cover. Single scene, not a time series.",
            )))
    except Exception as exc:  # noqa: BLE001
        bundle["satellite"] = {"error": f"Satellite fetch raised {type(exc).__name__}: {exc}"}

    try:
        fires = fetch_active_fires(lat, lon, radius_km=radius_km, days=5)
        bundle["active_fires"] = fires
        if fires.get("available"):
            evidence_records.append(_record_with_hash(EvidenceRecord.satellite(
                source=fires.get("source", "NASA FIRMS"),
                status="OBSERVED",
                temporal="OBSERVED",
                dataset="NASA FIRMS VIIRS/MODIS active fires",
                resolution=fires.get("resolution"),
                method=f"active-fire detection within {radius_km} km / {fires.get('days', 5)} days",
                location={"lat": lat, "lon": lon},
                limitations="Hotspot detections are points, not fire perimeters; confidence varies.",
            )))
    except Exception as exc:  # noqa: BLE001
        bundle["active_fires"] = {"error": f"Active-fires fetch raised {type(exc).__name__}: {exc}"}

    try:
        forest_loss = fetch_forest_loss(lat, lon)
        bundle["forest_loss"] = forest_loss
        if "error" not in forest_loss:
            evidence_records.append(_record_with_hash(EvidenceRecord.open_data(
                source=forest_loss.get("source", _FOREST_LOSS_PRODUCT),
                status="OBSERVED",
                temporal="HISTORICAL",
                dataset=forest_loss.get("dataset", _FOREST_LOSS_PRODUCT),
                resolution=forest_loss.get("resolution"),
                method="Hansen/UMD GFC windowed screening (~1 km box)",
                location={"lat": lat, "lon": lon},
                limitations=" ".join(forest_loss.get("limitations", [])) or "30 m resolution; 30% canopy threshold; screening only.",
            )))
    except Exception as exc:  # noqa: BLE001
        bundle["forest_loss"] = {"error": f"Forest-loss fetch raised {type(exc).__name__}: {exc}"}

    return {"bundle": bundle, "evidence_records": evidence_records}


def _evidence_id_by_source(evidence_records: List[Dict[str, Any]], source_substring: str) -> Optional[str]:
    for rec in evidence_records:
        source = (rec.get("source") or "").lower()
        if source_substring.lower() in source:
            return rec.get("evidence_id")
    return None


def _check_site_forested(
    bundle: Dict[str, Any],
    evidence_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    landcover = bundle.get("landcover") or {}
    if "error" in landcover or not landcover.get("dominant_label"):
        return {
            "check": "site_forested",
            "claim_type": "site_forested",
            "result": "cannot_assess",
            "basis": "Land-cover data is unavailable; forest status cannot be assessed.",
            "evidence_ids": [eid for eid in [_evidence_id_by_source(evidence_records, "worldcover")] if eid],
            "caveats": ["Land cover is a single-year snapshot, not a forest-loss time series."],
        }

    label = str(landcover.get("dominant_label", "")).lower()
    is_tree = "tree" in label
    evidence_ids = [eid for eid in [
        _evidence_id_by_source(evidence_records, "worldcover"),
        _evidence_id_by_source(evidence_records, "sentinel"),
    ] if eid]

    if is_tree:
        return {
            "check": "site_forested",
            "claim_type": "site_forested",
            "result": "consistent",
            "basis": f"Dominant land cover is '{landcover.get('dominant_label')}' (fraction {landcover.get('dominant_fraction')}).",
            "evidence_ids": evidence_ids,
            "caveats": [
                "Land cover is a single-year snapshot; it cannot prove the site was never cleared.",
            ],
        }

    return {
        "check": "site_forested",
        "claim_type": "site_forested",
        "result": "inconsistent",
        "basis": f"Dominant land cover is '{landcover.get('dominant_label')}' (fraction {landcover.get('dominant_fraction')}), not tree cover.",
        "evidence_ids": evidence_ids,
        "caveats": [
            "Land cover is a single-year snapshot; the observed class is inconsistent with the claim at this time.",
        ],
    }


def _check_no_recent_clearing(
    bundle: Dict[str, Any],
    evidence_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    forest_loss = bundle.get("forest_loss") or {}
    evidence_ids = [eid for eid in [
        _evidence_id_by_source(evidence_records, "hansen"),
        _evidence_id_by_source(evidence_records, "umd"),
    ] if eid]

    if "error" in forest_loss or forest_loss.get("loss_detected") is None:
        reason = forest_loss.get("error") or "Forest-loss data unavailable"
        return {
            "check": "no_recent_clearing",
            "claim_type": "no_recent_clearing",
            "result": "cannot_assess",
            "basis": f"Forest-loss layer is unavailable: {reason}.",
            "evidence_ids": evidence_ids,
            "caveats": ["GFC 2023 v1.11 covers 2001–2023 only."],
        }

    if forest_loss.get("loss_after_2020"):
        years = ", ".join(str(y) for y in sorted((forest_loss.get("loss_years") or {}).keys()) if y >= 2021)
        return {
            "check": "no_recent_clearing",
            "claim_type": "no_recent_clearing",
            "result": "inconsistent",
            "basis": f"Hansen/UMD GFC detects tree-cover loss in {years} after the 2020-12-31 cutoff.",
            "evidence_ids": evidence_ids,
            "caveats": [
                "30 m resolution; small clearings and degradation may be missed.",
                "GFC 2023 v1.11 covers through 2023; 2024+ loss is not included.",
            ],
        }

    if forest_loss.get("loss_detected"):
        latest = forest_loss.get("latest_loss_year")
        return {
            "check": "no_recent_clearing",
            "claim_type": "no_recent_clearing",
            "result": "consistent",
            "basis": f"Tree-cover loss detected only in {latest}, before the 2020-12-31 cutoff; no post-cutoff loss through 2023.",
            "evidence_ids": evidence_ids,
            "caveats": [
                "GFC 2023 v1.11 covers through 2023; 2024+ loss is not included.",
            ],
        }

    return {
        "check": "no_recent_clearing",
        "claim_type": "no_recent_clearing",
        "result": "consistent",
        "basis": "No Hansen/UMD GFC tree-cover loss detected in the screened window through 2023.",
        "evidence_ids": evidence_ids,
        "caveats": [
            "30 m resolution; small clearings and degradation may be missed.",
            "GFC 2023 v1.11 covers through 2023; 2024+ loss is not included.",
        ],
    }


def _check_no_burning(
    bundle: Dict[str, Any],
    evidence_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fires = bundle.get("active_fires") or {}
    if "error" in fires or not fires.get("available"):
        reason = fires.get("error") or "Active-fire data unavailable"
        return {
            "check": "no_burning",
            "claim_type": "no_burning",
            "result": "cannot_assess",
            "basis": f"Active-fire layer is unavailable: {reason}.",
            "evidence_ids": [eid for eid in [_evidence_id_by_source(evidence_records, "firms")] if eid],
            "caveats": [],
        }

    count = int(fires.get("count", 0))
    days = int(fires.get("days", 5))
    radius = fires.get("radius_km", "unknown")
    sensor = fires.get("sensor", "unknown")
    evidence_ids = [eid for eid in [_evidence_id_by_source(evidence_records, "firms")] if eid]

    if count == 0:
        return {
            "check": "no_burning",
            "claim_type": "no_burning",
            "result": "consistent",
            "basis": f"No active-fire detections within {radius} km in the last {days} days ({sensor}).",
            "evidence_ids": evidence_ids,
            "caveats": [
                "Small or low-temperature fires may be undetected; coverage is limited to the sensor window.",
            ],
        }

    return {
        "check": "no_burning",
        "claim_type": "no_burning",
        "result": "inconsistent",
        "basis": f"{count} active-fire detection(s) within {radius} km in the last {days} days ({sensor}).",
        "evidence_ids": evidence_ids,
        "caveats": [
            "Detections are hotspot points, not fire perimeters; some may be outside the exact site.",
        ],
    }


def _check_vegetation_present(
    bundle: Dict[str, Any],
    evidence_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    satellite = bundle.get("satellite") or {}
    evidence_ids = [eid for eid in [_evidence_id_by_source(evidence_records, "sentinel")] if eid]

    if "error" in satellite or satellite.get("ndvi") is None:
        return {
            "check": "vegetation_present",
            "claim_type": "vegetation_present",
            "result": "cannot_assess",
            "basis": "Sentinel-2 NDVI is unavailable; vegetation status cannot be assessed.",
            "evidence_ids": evidence_ids,
            "caveats": [],
        }

    ndvi = float(satellite["ndvi"])
    if ndvi >= 0.3:
        result = "consistent"
        basis = f"Sentinel-2 NDVI {ndvi:.3f} indicates active photosynthetic vegetation."
    elif ndvi < 0.2:
        result = "inconsistent"
        basis = f"Sentinel-2 NDVI {ndvi:.3f} is low; little active vegetation detected."
    else:
        result = "cannot_assess"
        basis = f"Sentinel-2 NDVI {ndvi:.3f} is ambiguous for the claim."

    return {
        "check": "vegetation_present",
        "claim_type": "vegetation_present",
        "result": result,
        "basis": basis,
        "evidence_ids": evidence_ids,
        "caveats": ["NDVI is scene-specific and affected by cloud, soil and seasonal conditions."],
    }


def _check_free_text() -> Dict[str, Any]:
    return {
        "check": "free_text",
        "claim_type": "free_text",
        "result": "cannot_assess",
        "basis": "No structured check exists for a free-text claim; the observed evidence bundle is provided for investigator assessment.",
        "evidence_ids": [],
        "caveats": [],
    }


def _run_consistency_checks(
    claim_type: str,
    bundle: Dict[str, Any],
    evidence_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if claim_type == "site_forested":
        return [_check_site_forested(bundle, evidence_records)]
    if claim_type == "no_recent_clearing":
        return [_check_no_recent_clearing(bundle, evidence_records)]
    if claim_type == "no_burning":
        return [_check_no_burning(bundle, evidence_records)]
    if claim_type == "vegetation_present":
        return [_check_vegetation_present(bundle, evidence_records)]
    if claim_type == "free_text":
        return [_check_free_text()]
    return []


def _build_declared_gaps(
    typology: str,
    bundle: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = [
        {
            "type": "vintage_limitation",
            "dataset": _FOREST_LOSS_PRODUCT,
            "reason": _FOREST_LOSS_VINTAGE_LIMITATION,
        },
        {
            "type": "financial_data",
            "dataset": None,
            "reason": _FINANCIAL_DATA_BOUNDARY,
        },
    ]

    if typology in ("illegal_mining", "waste_dumping"):
        gaps.append({
            "type": "dataset_not_integrated",
            "dataset": "Mining / waste disturbance detection layer",
            "reason": f"No dedicated {typology.replace('_', ' ')} detection dataset is integrated in this deployment.",
        })

    if typology == "unlicensed_clearing":
        gaps.append({
            "type": "dataset_not_integrated",
            "dataset": "Permit / land-title register",
            "reason": "Clearing legality requires permit status, which is not integrated.",
        })

    forest_loss = bundle.get("forest_loss") or {}
    if "error" in forest_loss:
        gaps.append({
            "type": "data_unavailable",
            "dataset": _FOREST_LOSS_PRODUCT,
            "reason": f"Forest-loss fetch failed: {forest_loss['error']}",
        })

    landcover = bundle.get("landcover") or {}
    if "error" in landcover:
        gaps.append({
            "type": "data_unavailable",
            "dataset": "ESA WorldCover",
            "reason": f"Land-cover fetch failed: {landcover['error']}",
        })

    satellite = bundle.get("satellite") or {}
    if "error" in satellite:
        gaps.append({
            "type": "data_unavailable",
            "dataset": "Sentinel-2 L2A",
            "reason": f"Sentinel-2 fetch failed: {satellite['error']}",
        })

    fires = bundle.get("active_fires") or {}
    if "error" in fires or not fires.get("available"):
        reason = fires.get("error") if isinstance(fires, dict) else "Active-fire data unavailable"
        gaps.append({
            "type": "data_unavailable",
            "dataset": "NASA FIRMS",
            "reason": f"Active-fire fetch failed: {reason}",
        })

    return gaps


def _case_verdict(checks: List[Dict[str, Any]]) -> str:
    if any(c["result"] == "inconsistent" for c in checks):
        return "inconsistencies_found"
    if any(c["result"] == "cannot_assess" for c in checks):
        return "partially_assessable"
    return "no_inconsistency_detected_with_current_evidence"


def assess_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an Environmental Forensic Evidence Pack.

    Input schema:
        {
          "title": "optional string",
          "typology": "illegal_logging|illegal_mining|unlicensed_clearing|waste_dumping|other",
          "site": {"name?": "...", "lat": ..., "lon": ...} | {"address": "..."},
          "subject_claim": {"type": "site_forested|no_recent_clearing|no_burning|vegetation_present|free_text", "text?": "..."},
          "reference_documents?": [{"title": "...", "url": "..."}],
          "radius_km?": 25
        }

    Raises ValueError for invalid typology/claim type or site.
    """
    title = str(case.get("title") or "").strip() or None
    typology_id = case.get("typology")
    typology = _typology_by_id(typology_id)
    if typology is None:
        valid = ", ".join(t["id"] for t in CASE_TYPOLOGIES)
        raise ValueError(f"typology must be one of: {valid}")

    subject_claim_raw = case.get("subject_claim")
    if not isinstance(subject_claim_raw, dict):
        raise ValueError("subject_claim must be an object")
    claim_type_id = subject_claim_raw.get("type")
    claim_type = _claim_type_by_id(claim_type_id)
    if claim_type is None:
        valid = ", ".join(c["id"] for c in CLAIM_TYPES)
        raise ValueError(f"subject_claim.type must be one of: {valid}")

    site = _resolve_site(case.get("site"))
    if "error" in site:
        raise ValueError(site["error"])

    radius_raw = case.get("radius_km", 25.0)
    try:
        radius_km = float(radius_raw)
    except (TypeError, ValueError):
        raise ValueError("radius_km must be a number")
    if not (1.0 <= radius_km <= 200.0):
        raise ValueError("radius_km must be between 1 and 200")

    reference_documents = _normalise_reference_documents(case.get("reference_documents"))

    fetched = _fetch_evidence_bundle(site["lat"], site["lon"], radius_km=radius_km)
    bundle = fetched["bundle"]
    evidence_records = fetched["evidence_records"]

    checks = _run_consistency_checks(claim_type_id, bundle, evidence_records)
    verdict = _case_verdict(checks)

    declared_gaps = _build_declared_gaps(typology_id, bundle)

    case_basis = {
        "title": title,
        "typology": typology_id,
        "site": {"name": site.get("name"), "lat": site["lat"], "lon": site["lon"]},
        "subject_claim": {"type": claim_type_id, "text": subject_claim_raw.get("text")},
        "reference_documents": reference_documents,
        "radius_km": radius_km,
    }
    case_id = content_hash(case_basis)[:16]

    chain_of_custody = {
        "case_id": case_id,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "evidence_records": [
            {
                "evidence_id": rec.get("evidence_id"),
                "source": rec.get("source"),
                "dataset": rec.get("dataset"),
                "acquired_at": rec.get("acquired_at"),
                "content_hash": rec.get("content_hash"),
            }
            for rec in evidence_records
        ],
        "note": (
            "Every evidence record is content-hashed at acquisition; records can be "
            "re-derived from the cited public sources."
        ),
    }

    return {
        "case_id": case_id,
        "title": title,
        "generated_at": utcnow_iso(),
        "engine_version": ENGINE_VERSION,
        "typology": {
            "id": typology["id"],
            "label": typology["label"],
            "note": typology["note"],
        },
        "site": site,
        "subject_claim": {
            "type": claim_type["id"],
            "label": claim_type["label"],
            "text": (subject_claim_raw.get("text") or "").strip() or None,
            "submitter_note": "Declared by the submitter — not verified by Talaix.",
        },
        "reference_documents": reference_documents,
        "radius_km": radius_km,
        "evidence_bundle": bundle,
        "checks": checks,
        "case_verdict": verdict,
        "verdict_note": LEGAL_NOTE,
        "chain_of_custody": chain_of_custody,
        "declared_gaps": declared_gaps,
        "frameworks": FORENSIC_FRAMEWORKS,
        "legal_note": LEGAL_NOTE,
        "disclaimer": FORENSICS_DISCLAIMER,
        "honesty_contract": HONESTY_CONTRACT,
        "authenticity": issue_seal("forensics", case_id, case_basis),
    }
