"""
Cascading Risk Graph v1 (IPCC AR6 WG2 compound/cascading risk framing —
``ipccar6wg2``; typology ``zscheischler2020typology``).

Combines:

- the curated structural knowledge graph ``config/cascading_graph.json``
  (hazard -> system disruption and system -> system propagation edges,
  reference-class knowledge, every edge carrying an evidence block —
  class / basis / confidence / limitations — and ``quantified: false``),
- the location's REAL active hazards (the light signal extraction of
  ``src/climate/compound.py`` — imported, not duplicated), and
- REAL exposure anchors from ``exposure_econ.build_economic_exposure``
  (mapped OSM/WorldCover counts).

Output: the cascade paths RELEVANT at this location — paths whose hazard is
currently elevated AND whose directly-exposed (entry) system has real mapped
anchors (value > 0 from the exposure block). Every path carries its edges,
mechanisms, per-edge evidence blocks, real anchor values and the explicit
statement:

    "Propagation likelihoods and losses are NOT quantified — this is a
    structural relevance graph, not a loss model."

Honesty contract:

- No propagation probabilities, no losses, no numeric cascade score.
- Honest empty state when no hazard is elevated.
- Honest insufficient-exposure note when OSM anchors are missing/failed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from ..dashboard.cache import cached
from .compound import extract_light_signals
from .evidence import utcnow_iso
from .exposure_econ import build_economic_exposure
from .ontology import Confidence, EvidenceClass

TTL_CASCADING = 3600.0  # 1 h

_DEFAULT_GRAPH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "cascading_graph.json"
)

NOT_QUANTIFIED_CASCADE_STATEMENT = (
    "Propagation likelihoods and losses are NOT quantified — this is a "
    "structural relevance graph, not a loss model."
)

_NO_ACTIVE_HAZARDS_STATEMENT = (
    "No hazard is currently elevated at this location, so no cascade paths "
    "are locally relevant. This is a structural relevance screening — "
    "absence of a path is not evidence of absence of risk."
)

_INSUFFICIENT_EXPOSURE_STATEMENT = (
    "Exposure anchors are unavailable or all unmapped here (OpenStreetMap "
    "anchors missing); cascade paths cannot be locally anchored."
)

_MAX_PATH_EDGES = 3        # hazard -> system -> system -> system
_MAX_PATHS_PER_HAZARD = 40


# ---------------------------------------------------------------------------
# Graph loading / validation
# ---------------------------------------------------------------------------


def load_cascading_graph(path: Optional[str] = None) -> Dict[str, Any]:
    graph_path = path or os.environ.get("HYDRASHIELD_CASCADING_GRAPH") or _DEFAULT_GRAPH
    with open(graph_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_cascading_graph(graph: Dict[str, Any]) -> List[str]:
    """Structural validation; returns a list of problems (empty = valid)."""

    problems: List[str] = []
    nodes = graph.get("nodes") or {}
    hazards = nodes.get("hazards") or {}
    systems = nodes.get("systems") or {}
    known = set(hazards) | set(systems)
    if not hazards:
        problems.append("no hazard nodes declared")
    if not systems:
        problems.append("no system nodes declared")
    valid_classes = {EvidenceClass.SCIENTIFIC.value, EvidenceClass.OPEN_DATA_OFFICIAL.value}
    valid_confidences = {Confidence.HIGH.value, Confidence.MEDIUM.value,
                         Confidence.LOW.value}
    for i, edge in enumerate(graph.get("edges") or []):
        frm, to = edge.get("from"), edge.get("to")
        if frm not in known:
            problems.append(f"edge {i}: unknown 'from' node '{frm}'")
        if to not in known:
            problems.append(f"edge {i}: unknown 'to' node '{to}'")
        if frm in hazards and to in hazards:
            problems.append(f"edge {i}: hazard->hazard edges are not modelled ({frm}->{to})")
        if not edge.get("mechanism"):
            problems.append(f"edge {i}: missing mechanism")
        if edge.get("evidence_class") not in valid_classes:
            problems.append(f"edge {i}: evidence_class must be one of {sorted(valid_classes)}")
        if edge.get("quantified") is not False:
            problems.append(f"edge {i}: quantified must be false (no invented metrics)")
        evidence = edge.get("evidence")
        if not isinstance(evidence, dict):
            problems.append(f"edge {i}: missing evidence block")
        else:
            if evidence.get("class") not in valid_classes:
                problems.append(
                    f"edge {i}: evidence.class must be one of {sorted(valid_classes)}")
            elif evidence.get("class") != edge.get("evidence_class"):
                problems.append(
                    f"edge {i}: evidence.class must match the edge evidence_class")
            if not evidence.get("basis"):
                problems.append(f"edge {i}: evidence.basis missing (documented "
                                "mechanism class, one line)")
            if evidence.get("confidence") not in valid_confidences:
                problems.append(
                    f"edge {i}: evidence.confidence must be one of "
                    f"{sorted(valid_confidences)}")
            if not evidence.get("limitations"):
                problems.append(f"edge {i}: evidence.limitations missing")
    # Anchor specs must point at exposure categories that exist.
    for sid, spec in systems.items():
        anchor = (spec or {}).get("anchor")
        if anchor is not None and not anchor.get("category"):
            problems.append(f"system '{sid}': anchor without category")
    return problems


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def _resolve_anchor(system_id: str, anchor_spec: Optional[Dict[str, Any]],
                    exposure_categories: Dict[str, Any]) -> Dict[str, Any]:
    """Map one system node to its REAL exposure anchor (or honest absence)."""

    if not anchor_spec:
        return {
            "system": system_id,
            "status": "no_anchor",
            "value": None,
            "unit": None,
            "note": "No mapped exposure anchor exists for this system in the "
                    "current exposure layer (declared data gap).",
        }
    category = exposure_categories.get(anchor_spec.get("category")) or {}
    value: Any = category
    for part in str(anchor_spec.get("field") or "").split("."):
        value = value.get(part) if isinstance(value, dict) else None
    unit = anchor_spec.get("unit", "count")
    if value is None:
        return {
            "system": system_id,
            "status": "not_mapped",
            "value": None,
            "unit": unit,
            "reason": category.get("reason") or f"exposure category "
                      f"'{anchor_spec.get('category')}' not mapped here",
            "source": category.get("source"),
        }
    return {
        "system": system_id,
        "status": "mapped",
        "value": value,
        "unit": unit,
        "source": category.get("source"),
        "completeness_caveat": category.get("completeness_caveat"),
    }


def _enumerate_paths(graph: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """All simple edge-paths starting at a hazard node (depth-capped)."""

    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for edge in graph.get("edges") or []:
        adjacency.setdefault(edge["from"], []).append(edge)

    paths: List[List[Dict[str, Any]]] = []

    def walk(node: str, trail: List[Dict[str, Any]], seen: set) -> None:
        if trail:
            paths.append(list(trail))
        if len(trail) >= _MAX_PATH_EDGES:
            return
        for edge in adjacency.get(node, []):
            nxt = edge["to"]
            if nxt in seen:
                continue
            walk(nxt, trail + [edge], seen | {nxt})

    for hazard in (graph.get("nodes", {}).get("hazards") or {}):
        collected_before = len(paths)
        walk(hazard, [], {hazard})
        # Depth cap safety: keep per-hazard output bounded and deterministic.
        hazard_paths = paths[collected_before:]
        if len(hazard_paths) > _MAX_PATHS_PER_HAZARD:
            del paths[collected_before:]
            paths.extend(hazard_paths[:_MAX_PATHS_PER_HAZARD])
    return paths


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@cached("cascading_assess", TTL_CASCADING)
def assess_cascading(
    lat: float,
    lon: float,
    *,
    active_hazards: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Cascading-risk relevance assessment for a point. Cached 1 h.

    ``active_hazards`` optionally overrides detection with caller-declared
    hazard ids (basis declared in the output); by default the location's
    real elevated hazards come from compound.extract_light_signals.
    """

    lat, lon = float(lat), float(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Coordinates out of range"}

    graph = load_cascading_graph()
    systems = (graph.get("nodes") or {}).get("systems") or {}
    hazard_nodes = (graph.get("nodes") or {}).get("hazards") or {}

    # -- real active hazards ------------------------------------------------
    hazards_unavailable: List[Dict[str, str]] = []
    signal_snapshot: Dict[str, Any] = {}
    if active_hazards is not None:
        active: List[str] = []
        unknown: List[str] = []
        for hid in {str(h).strip().lower() for h in active_hazards if str(h).strip()}:
            if hid in hazard_nodes:
                active.append(hid)
            else:
                unknown.append(hid)
        active_basis = "caller-declared active hazards (no level computed on this path)"
        if unknown:
            hazards_unavailable.append({
                "hazard": ",".join(sorted(unknown)),
                "reason": "caller-declared hazard(s) not in the cascading graph",
            })
    else:
        extracted = extract_light_signals(lat, lon)
        hazards_unavailable = list(extracted.get("hazards_unavailable") or [])
        for hid, sig in (extracted.get("signals") or {}).items():
            signal_snapshot[hid] = {
                "elevated": bool(sig.get("elevated")),
                "elevated_basis": sig.get("elevated_basis"),
                "level": sig.get("level"),
                "values": sig.get("values"),
                "summary": sig.get("summary"),
            }
        active = sorted(hid for hid, s in signal_snapshot.items() if s["elevated"])
        active_basis = "real elevated hazards from the compound light signal extraction"

    # -- real exposure anchors ----------------------------------------------
    exposure = build_economic_exposure(lat, lon)
    exposure_categories = exposure.get("exposure") or {}
    anchors = {
        sid: _resolve_anchor(sid, (spec or {}).get("anchor"), exposure_categories)
        for sid, spec in systems.items()
    }
    exposure_failed = "error" in exposure
    any_anchor_mapped = any(a["status"] == "mapped" for a in anchors.values())
    exposure_status = "ok" if (not exposure_failed and any_anchor_mapped) else "insufficient"

    # -- relevant cascade paths ----------------------------------------------
    cascade_paths: List[Dict[str, Any]] = []
    if exposure_status == "ok" and active:
        for trail in _enumerate_paths(graph):
            hazard = trail[0]["from"]
            if hazard not in active:
                continue
            entry_system = trail[0]["to"]
            entry_anchor = anchors.get(entry_system) or {}
            # Relevance rule (declared): the directly-exposed system must have
            # a real mapped anchor value > 0 here.
            if not (entry_anchor.get("status") == "mapped"
                    and isinstance(entry_anchor.get("value"), (int, float))
                    and entry_anchor["value"] > 0):
                continue
            touched_systems = [trail[0]["to"]] + [e["to"] for e in trail[1:]]
            cascade_paths.append({
                "hazard": hazard,
                "hazard_basis": (signal_snapshot.get(hazard) or {}).get("elevated_basis")
                or active_basis,
                "nodes": [hazard] + touched_systems,
                "edges": [
                    {
                        "from": e["from"],
                        "to": e["to"],
                        "mechanism": e["mechanism"],
                        "evidence_class": e["evidence_class"],
                        "evidence": e.get("evidence"),
                        "quantified": False,
                    }
                    for e in trail
                ],
                "anchors": {sid: anchors[sid] for sid in touched_systems},
                "fully_anchored": all(
                    anchors[sid]["status"] == "mapped" for sid in touched_systems
                ),
                "not_quantified_statement": NOT_QUANTIFIED_CASCADE_STATEMENT,
            })
        cascade_paths.sort(key=lambda p: (p["hazard"], len(p["nodes"]), p["nodes"]))

    # -- honest states --------------------------------------------------------
    no_cascade_signal = None
    if not active:
        no_cascade_signal = {
            "status": "no_active_hazards",
            "statement": _NO_ACTIVE_HAZARDS_STATEMENT,
        }
    elif exposure_status != "ok":
        no_cascade_signal = {
            "status": "insufficient_exposure",
            "statement": _INSUFFICIENT_EXPOSURE_STATEMENT,
        }
    elif not cascade_paths:
        no_cascade_signal = {
            "status": "no_anchored_paths",
            "statement": (
                "Hazards are elevated but none of the cascade paths' directly-"
                "exposed systems have mapped anchors with value > 0 here; "
                "paths are not reported without local anchors."
            ),
        }

    status = "ok" if cascade_paths else ("partial" if not exposure_failed else "unavailable")

    return {
        "status": status,
        "location": {"lat": lat, "lon": lon},
        "generated_at": utcnow_iso(),
        "active_hazards": [
            {
                "hazard": hid,
                "elevated": True,
                "basis": (signal_snapshot.get(hid) or {}).get("elevated_basis")
                or active_basis,
                "level": (signal_snapshot.get(hid) or {}).get("level"),
            }
            for hid in active
        ],
        "active_hazards_basis": active_basis,
        "hazards_unavailable": hazards_unavailable,
        "cascade_paths": cascade_paths,
        "cascade_path_count": len(cascade_paths),
        "no_cascade_signal": no_cascade_signal,
        "exposure_status": exposure_status,
        "exposure_note": None if exposure_status == "ok" else (
            exposure.get("error") or _INSUFFICIENT_EXPOSURE_STATEMENT
        ),
        "relevance_rule": (
            "A path is relevant when its hazard is currently elevated here AND "
            "its directly-exposed (entry) system has a real mapped anchor "
            "value > 0 from the exposure block. Downstream nodes report their "
            "own anchors honestly (mapped / not_mapped / no_anchor)."
        ),
        "not_quantified_statement": NOT_QUANTIFIED_CASCADE_STATEMENT,
        "uncertainty": {
            "confidence": Confidence.LOW.value,
            "note": "Structural relevance screening over a curated graph and "
                    "mapped exposure anchors; not a propagation or loss model.",
            "sources_of_uncertainty": [
                "OpenStreetMap completeness varies by region; anchor counts are "
                "a lower bound",
                "Reference-class edge mechanisms; local system dependencies may "
                "differ",
                "Active hazards are screening-level reanalysis signals",
            ],
        },
        "limitations": [
            NOT_QUANTIFIED_CASCADE_STATEMENT,
            "Paths stop at depth 3 edges; cyclic propagation is not modelled.",
            "Systems without a mapped anchor (rail, ports where unmapped, "
            "logistics, telecom, business_continuity) are reported with "
            "anchor status no_anchor/not_mapped — never invented counts.",
        ],
        "provenance": {
            "engine": "cascading_risk_graph_v1",
            "graph_id": graph.get("graph_id"),
            "graph_version": graph.get("version"),
            "graph_config": "config/cascading_graph.json",
            "research": [
                {"id": "ipccar6wg2",
                 "role": "risk framework: hazard x exposure x vulnerability; "
                         "compound/cascading risk"},
                {"id": "zscheischler2020typology",
                 "role": "compound-event typology context"},
            ],
            "exposure_source": "src/climate/exposure_econ.py (OSM/ohsome + "
                               "Overpass counts, ESA WorldCover)",
            "hazard_source": "src/climate/compound.py light signal extraction",
        },
    }
