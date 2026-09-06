"""
Offline tests for the climate ontology and evidence core (Stage 1).

No network, no fixtures on disk — pure vocabulary/contract tests.
"""

import pytest

from src.climate.ontology import (
    CAUSE_STATUSES,
    ClaimStatus,
    Confidence,
    EvidenceClass,
    Exposure,
    HazardType,
    Impact,
    Observation,
    TemporalClass,
    Uncertainty,
    validate_cause,
)
from src.climate.evidence import (
    EvidenceRecord,
    content_hash,
    upgrade_legacy_provenance,
    upgrade_provenance_block,
)


# ---------------------------------------------------------------------------
# Ontology vocabularies
# ---------------------------------------------------------------------------


def test_hazard_types_are_the_registered_set():
    assert {h.value for h in HazardType} == {
        "wildfire",
        "flood",
        "drought",
        "heat",
        "wind",
        "coastal",
        # geophysical hazard — wired 2026-09 (USGS ComCat + EMSC)
        "earthquake",
        # expansion candidates (registered with honest unavailable states)
        "dust",
        "volcanic",
    }


def test_claim_status_vocabulary():
    assert {s.value for s in ClaimStatus} == {
        "OBSERVED",
        "DOCUMENTED",
        "REPORTED",
        "MODELLED",
        "INFERRED",
        "UNKNOWN",
    }


def test_temporal_vocabulary_separates_projections():
    values = {t.value for t in TemporalClass}
    assert {"OBSERVED", "HISTORICAL", "FORECAST", "PROJECTED", "SCENARIO"} == values


def test_evidence_classes_are_exactly_five():
    assert {e.value for e in EvidenceClass} == {
        "SCIENTIFIC",
        "SATELLITE_EO",
        "OPEN_DATA_OFFICIAL",
        "MEDIA",
        "MODELLED",
    }


def test_cause_discipline_only_documented_or_unknown():
    assert CAUSE_STATUSES == {"DOCUMENTED", "UNKNOWN"}
    assert validate_cause("DOCUMENTED") == "DOCUMENTED"
    assert validate_cause("UNKNOWN") == "UNKNOWN"
    # Media, models and inference can never establish a cause:
    assert validate_cause("REPORTED") == "UNKNOWN"
    assert validate_cause("MODELLED") == "UNKNOWN"
    assert validate_cause("INFERRED") == "UNKNOWN"
    assert validate_cause("OBSERVED") == "UNKNOWN"


def test_observation_defaults_and_serialisation():
    obs = Observation(quantity="river_discharge", value=312.4, unit="m³/s")
    d = obs.to_dict()
    assert d["status"] == "OBSERVED"
    assert d["temporal"] == "OBSERVED"
    assert d["quantity"] == "river_discharge"


def test_exposure_allows_not_quantified():
    exp = Exposure(category="buildings", count=None, description="not quantified")
    assert exp.count is None
    assert exp.to_dict()["status"] == "OBSERVED"


def test_impact_defaults_to_unknown():
    imp = Impact(kind="burned_area_ha")
    assert imp.status == "UNKNOWN"


def test_uncertainty_serialises():
    u = Uncertainty(note="grid smoothing", confidence=Confidence.LOW.value)
    assert u.to_dict()["confidence"] == "low"


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------


def test_evidence_record_requires_valid_vocabularies():
    with pytest.raises(ValueError):
        EvidenceRecord("NOT_A_CLASS", "OBSERVED", "OBSERVED", "source")
    with pytest.raises(ValueError):
        EvidenceRecord("MEDIA", "NOT_A_STATUS", "OBSERVED", "source")
    with pytest.raises(ValueError):
        EvidenceRecord("MEDIA", "OBSERVED", "NOT_A_TEMPORAL", "source")


def test_evidence_record_id_is_stable_and_content_derived():
    rec = EvidenceRecord.open_data(
        "Open-Meteo archive (ERA5)",
        dataset="ERA5 daily",
        reference_period={"start": "2024-08-01", "end": "2024-08-15"},
    )
    same = EvidenceRecord.open_data(
        "Open-Meteo archive (ERA5)",
        dataset="ERA5 daily",
        reference_period={"start": "2024-08-01", "end": "2024-08-15"},
    )
    assert rec.id == same.id
    assert len(rec.id) == 16
    d = rec.to_dict()
    assert d["evidence_id"] == rec.id
    assert d["evidence_class"] == "OPEN_DATA_OFFICIAL"


def test_content_hash_binds_claim_to_bytes():
    payload = {"daily": {"tmax": [31.2, 33.1]}, "unit": "°C"}
    h1 = content_hash(payload)
    h2 = content_hash({"unit": "°C", "daily": {"tmax": [31.2, 33.1]}})
    assert h1 == h2  # key-order independent
    assert content_hash({"daily": {"tmax": [31.2]}}) != h1


def test_media_records_cannot_exceed_reported_status():
    rec = EvidenceRecord.media("Example News", link="https://example.com/a", status="OBSERVED")
    assert rec.claim_status == "REPORTED"
    assert rec.evidence_class == "MEDIA"


def test_unknown_record_carries_why():
    rec = EvidenceRecord.unknown("NASA FIRMS", why="FIRMS_MAP_KEY not configured")
    assert rec.claim_status == "UNKNOWN"
    assert "FIRMS_MAP_KEY" in rec.limitations
    assert rec.confidence == "low"


def test_modelled_record_requires_method():
    rec = EvidenceRecord.modelled("Talaix", method="SPI-style anomaly, declared window")
    assert rec.evidence_class == "MODELLED"
    assert rec.claim_status == "MODELLED"
    assert "SPI" in rec.method


def test_scientific_record_defaults_documented_historical():
    rec = EvidenceRecord.scientific("Van Wagner (1987), FWI System")
    assert rec.evidence_class == "SCIENTIFIC"
    assert rec.claim_status == "DOCUMENTED"
    assert rec.temporal == "HISTORICAL"


# ---------------------------------------------------------------------------
# Legacy provenance upgrade
# ---------------------------------------------------------------------------


def test_legacy_kind_alias_mapping():
    cases = {
        "observed": ("OBSERVED", "OBSERVED"),
        "derived": ("INFERRED", "OBSERVED"),
        "modeled": ("MODELLED", "OBSERVED"),
        "modelled": ("MODELLED", "OBSERVED"),
        "forecast": ("MODELLED", "FORECAST"),
        "unavailable": ("UNKNOWN", "OBSERVED"),
    }
    for kind, (status, temporal) in cases.items():
        out = upgrade_legacy_provenance({"kind": kind, "source": "Open-Meteo"})
        assert out["claim_status"] == status, kind
        assert out["temporal_class"] == temporal, kind


def test_legacy_modeled_spelling_is_normalised():
    out = upgrade_legacy_provenance({"kind": "modeled", "source": "Talaix models"})
    assert out["kind"] == "modelled"
    assert out["evidence_class"] == "MODELLED"


def test_unknown_legacy_kind_maps_to_unknown():
    out = upgrade_legacy_provenance({"kind": "guesstimate", "source": "nowhere"})
    assert out["claim_status"] == "UNKNOWN"


def test_evidence_class_inference_from_source_label():
    sat = upgrade_legacy_provenance({"kind": "observed", "source": "Copernicus Sentinel-2 (Element84 STAC)"})
    assert sat["evidence_class"] == "SATELLITE_EO"
    sci = upgrade_legacy_provenance({"kind": "derived", "source": "Van Wagner 1987 equations"})
    assert sci["evidence_class"] == "SCIENTIFIC"
    official = upgrade_legacy_provenance({"kind": "observed", "source": "Open-Meteo archive (ERA5)"})
    assert official["evidence_class"] == "OPEN_DATA_OFFICIAL"


def test_upgrade_provenance_block_preserves_shape_and_original_keys():
    block = {
        "weather": {"kind": "observed", "source": "Open-Meteo", "quality": "ok"},
        "risk_score": {"kind": "modeled", "source": "Talaix models"},
        "custom_note": "not-a-provenance-record",
    }
    out = upgrade_provenance_block(block)
    assert set(out) == {"weather", "risk_score", "custom_note"}
    assert out["weather"]["quality"] == "ok"  # original keys preserved
    assert out["weather"]["claim_status"] == "OBSERVED"
    assert out["risk_score"]["kind"] == "modelled"
    assert out["custom_note"] == "not-a-provenance-record"
