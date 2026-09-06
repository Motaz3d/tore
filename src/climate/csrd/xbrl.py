"""
XBRL / iXBRL rendering for CsrdTX assessments.

Produces the machine-readable output of a CSRD assessment:

- **XBRL 2.1 instance document** (``.xbrl``) — pure machine-readable.
- **Inline XBRL XHTML document** (``.xhtml``) — human-readable report
  with the same facts embedded as ``ix:`` tags (the ESEF pattern).

Honesty rules (same contract as the engine):

- Facts are emitted only for values that actually exist in the
  assessment. A missing value emits **no fact** and is listed in the
  tagging notes — never invented, never nil-padded.
- All facts use the documented Talaix extension namespace
  (``config/csrd/xbrl_mapping.json`` is the source of truth; the served
  taxonomy ``website/xbrl/csrd/2026/talaix-csrd.xsd`` mirrors it and a
  test keeps them in sync). Anchoring to the official ESRS digital
  taxonomy is a declared gap, stated in every output.
- Deterministic: same assessment object → same document bytes.

No I/O beyond the mapping/config files; no Flask imports.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from ..evidence import content_hash

_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "csrd", "xbrl_mapping.json"
)

_OFFICIAL_NOTE = (
    "Anchoring to the official ESRS digital (XBRL) taxonomy is a declared "
    "gap: all facts use the documented Talaix extension namespace. This is "
    "a machine-readable assessment extract, not an ESEF filing package."
)

_FORMATS = ("xbrl", "ixbrl")


def load_mapping() -> Dict[str, Any]:
    path = os.environ.get("HYDRASHIELD_CSRD_XBRL_MAPPING") or _MAPPING_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(path: str, root: Dict[str, Any]) -> Any:
    node: Any = root
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _fmt_number(value: Any) -> str:
    """Deterministic numeric rendering: ints without trailing .0."""
    if isinstance(value, bool):  # guard: bool is int subclass
        return "true" if value else "false"
    f = float(value)
    return str(int(f)) if f == int(f) else repr(f)


def collect_facts(assessment: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Collect (facts, tagging_notes) from an assessment.

    A fact is ``{name, value, type, unit}`` with ``value`` already
    rendered as a string. Every skipped element appends a tagging note
    stating why — the honesty channel of the XBRL output.
    """
    mapping = load_mapping()
    facts: List[Dict[str, Any]] = []
    notes: List[str] = []

    def emit(name: str, value: Any, type_: str, unit: Optional[str]) -> None:
        if value is None or value == "":
            notes.append(f"No fact emitted for {name}: value unavailable in the assessment.")
            return
        if type_ == "boolean":
            rendered = "true" if bool(value) else "false"
        elif type_ in ("integer", "monetary", "score"):
            try:
                rendered = _fmt_number(value)
            except (TypeError, ValueError):
                notes.append(f"No fact emitted for {name}: value is not numeric.")
                return
        else:
            rendered = str(value)
        facts.append({"name": name, "value": rendered, "type": type_, "unit": unit})

    for element in mapping["elements"]:
        emit(element["name"], _resolve(element["source"], assessment), element["type"], element.get("unit"))

    tp = mapping["topic_elements"]
    by_topic = {m.get("topic"): m for m in assessment.get("materiality") or []}
    for topic_id in tp["topics"]:
        entry = by_topic.get(topic_id)
        if not entry or entry.get("status") != "ASSESSED":
            notes.append(
                f"No facts emitted for topic {topic_id}: "
                f"{(entry or {}).get('reason') or 'not assessed'} — declared, never invented."
            )
            continue
        for suffix in tp["suffixes"]:
            name = tp["pattern"].replace("{ID}", topic_id).replace("{Suffix}", suffix["suffix"])
            emit(name, entry.get(suffix["source"]), suffix["type"], suffix.get("unit"))

    return facts, notes


def _context_block(assessment: Dict[str, Any]) -> Tuple[str, str]:
    """Return (context_id, context_xml) for the assessment's reporting year."""
    year = assessment.get("applicability", {}).get("reporting_year")
    fields = assessment.get("company", {}).get("fields", {})
    ctx_id = f"ctx-FY{year or 'unknown'}"
    lei = (fields.get("lei") or "").strip()
    if lei:
        scheme = "http://standards.iso.org/iso/17442"
        identifier = escape(lei)
    else:
        scheme = "https://talaix.com/entity"
        identifier = content_hash({"name": (fields.get("name") or "").strip().lower()})[:16]
    period = ""
    if year:
        period = (
            f"<period><startDate>{int(year):04d}-01-01</startDate>"
            f"<endDate>{int(year):04d}-12-31</endDate></period>"
        )
    else:
        period = "<period><instant>1970-01-01</instant></period>"
    xml = (
        f'<context id="{ctx_id}"><entity>'
        f'<identifier scheme="{scheme}">{identifier}</identifier>'
        f"</entity>{period}</context>"
    )
    return ctx_id, xml


def _units_block(mapping: Dict[str, Any]) -> str:
    return "".join(
        f'<unit id="{unit_id}"><measure>{measure}</measure></unit>'
        for unit_id, measure in mapping["units"].items()
    )


def _decimals(fact: Dict[str, Any]) -> str:
    return "0" if fact["type"] in ("integer", "monetary") else "INF"


def _header_comment(assessment: Dict[str, Any], notes: List[str]) -> str:
    lines = [
        f"CsrdTX assessment {assessment.get('assessment_id')} — engine "
        f"{assessment.get('engine_version')} — generated {assessment.get('generated_at')}",
        _OFFICIAL_NOTE,
        "Tagging notes (missing values are declared, never invented):",
        *["- " + n for n in notes],
    ]
    return "<!--\n" + "\n".join(lines) + "\n-->"


def build_xbrl_instance(assessment: Dict[str, Any]) -> str:
    """Render the assessment as an XBRL 2.1 instance document."""
    mapping = load_mapping()
    facts, notes = collect_facts(assessment)
    ctx_id, ctx_xml = _context_block(assessment)

    ns = mapping["namespace"]
    prefix = mapping["prefix"]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        _header_comment(assessment, notes),
        (
            f'<xbrl xmlns="http://www.xbrl.org/2003/instance" '
            f'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
            f'xmlns:link="http://www.xbrl.org/2003/linkbase" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'xmlns:iso4217="http://www.xbrl.org/2003/iso4217" '
            f'xmlns:{prefix}="{ns}">'
        ),
        (
            f'<link:schemaRef xlink:type="simple" '
            f'xlink:href="{mapping["taxonomy_url"]}"/>'
        ),
        ctx_xml,
        _units_block(mapping),
    ]
    for fact in facts:
        tag = f"{prefix}:{fact['name']}"
        attrs = f'contextRef="{ctx_id}"'
        if fact["unit"]:
            attrs += f' unitRef="{fact["unit"]}" decimals="{_decimals(fact)}"'
        parts.append(f"<{tag} {attrs}>{escape(fact['value'])}</{tag}>")
    parts.append("</xbrl>")
    return "\n".join(parts) + "\n"


def _ix_fact(fact: Dict[str, Any], ctx_id: str, prefix: str, display: Optional[str] = None) -> str:
    name = f"{prefix}:{fact['name']}"
    text = escape(display if display is not None else fact["value"])
    if fact["unit"]:
        return (
            f'<ix:nonFraction name="{name}" contextRef="{ctx_id}" '
            f'unitRef="{fact["unit"]}" decimals="{_decimals(fact)}">{text}</ix:nonFraction>'
        )
    return f'<ix:nonNumeric name="{name}" contextRef="{ctx_id}">{text}</ix:nonNumeric>'


def build_ixbrl_document(assessment: Dict[str, Any]) -> str:
    """Render the assessment as an inline-XBRL XHTML report."""
    mapping = load_mapping()
    facts, notes = collect_facts(assessment)
    ctx_id, ctx_xml = _context_block(assessment)
    prefix = mapping["prefix"]
    ns = mapping["namespace"]
    by_name = {f["name"]: f for f in facts}

    def row(label: str, element: str) -> str:
        fact = by_name.get(element)
        value = _ix_fact(fact, ctx_id, prefix) if fact else "—"
        return f"<tr><th>{escape(label)}</th><td>{value}</td></tr>"

    fields = assessment.get("company", {}).get("fields", {})
    applicability = assessment.get("applicability", {})
    readiness = assessment.get("readiness", {})

    topic_rows = []
    for m in assessment.get("materiality") or []:
        topic_id = m.get("topic")
        if m.get("status") == "ASSESSED":
            score_fact = by_name.get(f"Topic{topic_id}CombinedScore")
            mat_fact = by_name.get(f"Topic{topic_id}Material")
            conf_fact = by_name.get(f"Topic{topic_id}Confidence")
            topic_rows.append(
                "<tr><td>" + escape(topic_id) + "</td>"
                "<td>" + (_ix_fact(score_fact, ctx_id, prefix) if score_fact else "—") + "</td>"
                "<td>" + (_ix_fact(mat_fact, ctx_id, prefix) if mat_fact else "—") + "</td>"
                "<td>" + (_ix_fact(conf_fact, ctx_id, prefix) if conf_fact else "—") + "</td></tr>"
            )
        else:
            topic_rows.append(
                f"<tr><td>{escape(topic_id)}</td><td colspan=\"3\">"
                f"NOT ASSESSED — {escape(m.get('reason') or 'declared gap')}</td></tr>"
            )

    notes_html = "".join(f"<li>{escape(n)}</li>" for n in notes)
    title_name = escape(fields.get("name") or "Company")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:xbrli="http://www.w3.org/2003/instance"
      xmlns:link="http://www.xbrl.org/2003/linkbase"
      xmlns:xlink="http://www.w3.org/1999/xlink"
      xmlns:{prefix}="{ns}">
<head>
<meta charset="UTF-8" />
<title>CsrdTX assessment — {title_name}</title>
<ix:header>
  <ix:resources>
    {ctx_xml}
    {_units_block(mapping)}
  </ix:resources>
  <ix:references>
    <link:schemaRef xlink:type="simple" xlink:href="{mapping['taxonomy_url']}" />
  </ix:references>
</ix:header>
</head>
<body>
<h1>CSRD assessment — {title_name}</h1>
<p>Assessment {escape(str(assessment.get('assessment_id')))} · CsrdTX engine
{escape(str(assessment.get('engine_version')))} ·
{escape(str(assessment.get('generated_at')))}</p>

<h2>Entity profile</h2>
<table>
{row('Entity name', 'EntityName')}
{row('Country', 'EntityCountry')}
{row('Sector', 'EntitySector')}
{row('Employees', 'Employees')}
{row('Net turnover (EUR)', 'NetTurnoverEUR')}
{row('Balance sheet total (EUR)', 'BalanceSheetTotalEUR')}
{row('Listed', 'Listed')}
{row('Reporting year', 'ReportingYear')}
</table>

<h2>Applicability</h2>
<table>
{row('Determination', 'ApplicabilityDetermination')}
{row('Rule set', 'ApplicabilityRuleSet')}
{row('Wave', 'ApplicabilityWave')}
{row('ESRS version', 'EsrsVersion')}
{row('ESRS version status', 'EsrsVersionStatus')}
</table>

<h2>Readiness</h2>
<table>
{row('Overall (0–100)', 'ReadinessOverall')}
{row('Applicability clarity', 'ReadinessApplicabilityClarity')}
{row('Evidence coverage', 'ReadinessEvidenceCoverage')}
{row('Data completeness', 'ReadinessDataCompleteness')}
{row('Materiality readiness', 'ReadinessMaterialityReadiness')}
</table>

<h2>Double materiality</h2>
<table>
<thead><tr><th>Topic</th><th>Combined score (0–5)</th><th>Material</th><th>Confidence</th></tr></thead>
<tbody>
{''.join(topic_rows)}
</tbody>
</table>

<h2>Tagging notes</h2>
<p>{escape(_OFFICIAL_NOTE)}</p>
<ul>
{notes_html}
</ul>

<p><em>{escape(str(assessment.get('honesty_contract') or ''))}</em></p>
</body>
</html>
"""


def render(assessment: Dict[str, Any], fmt: str) -> str:
    """Render the assessment in ``fmt`` ('xbrl' or 'ixbrl')."""
    if fmt == "xbrl":
        return build_xbrl_instance(assessment)
    if fmt == "ixbrl":
        return build_ixbrl_document(assessment)
    raise ValueError(f"Unknown XBRL format '{fmt}' — expected one of {_FORMATS}")
